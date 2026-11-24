"""洞察工坊 · 综合分析看板订阅测试

验证看板取数入口与订阅发件入口发布范围一致：
- 订阅结果不计入草稿（未发布）金额；
- 同名查询的缓存按部门隔离，过期后按原部门范围重查；
- 重启/并发恢复同一次订阅只发一份完整结果，不跨版本混用。
"""
from __future__ import annotations

import threading

import pytest

from app.analysis_pipeline import (
    pipeline_dashboard_subscription,
    run_full_analysis_pipeline,
)
from app.dashboard_subscription import (
    DRAFT,
    PUBLISHED,
    ConfirmedQueryScope,
    DashboardQueryCache,
    DepartmentWalletLedger,
    SendRecord,
    SendRecordStore,
    SubscriptionService,
    WalletFact,
    execute_confirmed_query,
    fetch_dashboard_data,
    verify_send_record,
)

TTL = 300


# ============================================================
# 共享测试数据：两个部门的同名查询钱包
# ============================================================

@pytest.fixture
def ledger():
    """甲部：已发布 8 元 + 草稿 90 元；乙部：已发布 20 元。"""
    return DepartmentWalletLedger([
        WalletFact("F_A_PUB", "甲部", 8.0, PUBLISHED, 1),
        WalletFact("F_A_DRAFT", "甲部", 90.0, DRAFT, 1),
        WalletFact("F_B_PUB", "乙部", 20.0, PUBLISHED, 1),
    ])


@pytest.fixture
def scope_a():
    return ConfirmedQueryScope(query_name="部门钱包汇总", department="甲部")


@pytest.fixture
def scope_b():
    return ConfirmedQueryScope(query_name="部门钱包汇总", department="乙部")


@pytest.fixture
def cache():
    return DashboardQueryCache()


@pytest.fixture
def send_store():
    return SendRecordStore()


@pytest.fixture
def service(ledger, cache, send_store):
    return SubscriptionService(ledger, cache, send_store, ttl_ticks=TTL)


# ============================================================
# 看板查询：按发布状态过滤
# ============================================================

class TestDashboardQueryScope:
    def test_dashboard_query_excludes_draft_amounts(self, ledger, scope_a):
        result = execute_confirmed_query(scope_a, ledger)
        assert result.amount == 8.0  # 只计已发布，不含草稿 90
        assert [f.fact_id for f in result.facts] == ["F_A_PUB"]
        assert result.data_version == 1

    def test_confirmed_scope_defaults_to_published(self, scope_a, scope_b):
        assert scope_a.publish_status == PUBLISHED
        # 同名查询：查询名称相同，部门不同
        assert scope_a.query_name == scope_b.query_name == "部门钱包汇总"
        assert scope_a.cache_key() != scope_b.cache_key()

    def test_subscription_result_matches_dashboard_amount(
        self, ledger, cache, service, scope_a
    ):
        record = service.run_subscription("SUB_A", "SUB_A#1", scope_a, now=0)
        dashboard = fetch_dashboard_data(scope_a, ledger, cache, now=0)
        # 订阅发件金额与看板取数一致：8 元，不是计入草稿的 98 元
        assert record.amount == 8.0
        assert record.amount != 98.0
        assert record.amount == dashboard.amount

    def test_send_record_carries_scope_and_facts(self, service, scope_a):
        record = service.run_subscription("SUB_A", "SUB_A#1", scope_a, now=0)
        # 当次查询条件
        assert record.scope.query_name == "部门钱包汇总"
        assert record.scope.department == "甲部"
        assert record.scope.publish_status == PUBLISHED
        # 入算数据
        assert [f.fact_id for f in record.facts] == ["F_A_PUB"]
        assert all(f.status == PUBLISHED for f in record.facts)


# ============================================================
# 同名缓存：跨部门隔离
# ============================================================

class TestSameNameCacheIsolation:
    def test_dept_a_cache_never_hits_dept_b_or_draft_sum(
        self, ledger, cache, scope_a, scope_b
    ):
        fetch_dashboard_data(scope_b, ledger, cache, now=0)  # 乙部先写入缓存 20
        result_a = fetch_dashboard_data(scope_a, ledger, cache, now=0)
        assert result_a.amount == 8.0
        assert result_a.amount != 20.0  # 不得命中乙部缓存
        assert result_a.amount != 98.0  # 不得命中含草稿的口径
        # 两个部门各自持有独立缓存条目
        assert len(cache.keys()) == 2

    def test_cross_reading_same_name_queries(self, ledger, cache, scope_a, scope_b):
        # 另起两个同名查询交叉读缓存
        assert fetch_dashboard_data(scope_a, ledger, cache, now=0).amount == 8.0
        assert fetch_dashboard_data(scope_b, ledger, cache, now=1).amount == 20.0
        assert fetch_dashboard_data(scope_a, ledger, cache, now=2).amount == 8.0
        # 乙部仍为 20 元
        assert fetch_dashboard_data(scope_b, ledger, cache, now=3).amount == 20.0
        assert cache.hits == 2  # 后两次各自命中本部门缓存

    def test_expired_cache_requeries_same_department(
        self, ledger, cache, scope_a, scope_b
    ):
        fetch_dashboard_data(scope_a, ledger, cache, now=0)
        fetch_dashboard_data(scope_b, ledger, cache, now=0)
        misses_before = cache.misses
        # 甲部缓存过期后重查：仍按甲部范围取数，不得命中乙部的 20
        result = fetch_dashboard_data(scope_a, ledger, cache, now=TTL)
        assert result.amount == 8.0
        assert result.amount != 20.0
        assert result.from_cache is False
        assert cache.misses == misses_before + 1
        # 乙部过期后重查仍为 20
        assert fetch_dashboard_data(scope_b, ledger, cache, now=TTL).amount == 20.0


# ============================================================
# 重启恢复：exactly-once + 单版本一致
# ============================================================

class TestRestartRecovery:
    def test_concurrent_recovery_sends_one_consistent_record(
        self, ledger, cache, send_store, service, scope_a
    ):
        # 崩溃前：订阅任务已取得 8 元旧缓存，尚未写入发件记录
        stale = fetch_dashboard_data(scope_a, ledger, cache, now=0)
        assert stale.amount == 8.0
        # 一条甲部草稿发布：已发布数据升为 98 元（版本 2）
        ledger.publish("F_A_DRAFT")
        assert ledger.current_version("甲部") == 2

        # 重启：换新的服务实例，台账/缓存/发件存储是持久层
        recovered_service = SubscriptionService(ledger, cache, send_store, ttl_ticks=TTL)

        # 并发恢复同一次订阅
        results: list[SendRecord] = []
        barrier = threading.Barrier(2)

        def recover():
            barrier.wait()
            results.append(
                recovered_service.run_subscription("SUB_A", "SUB_A#run1", scope_a, now=1)
            )

        threads = [threading.Thread(target=recover) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 只发一份完整查询结果
        sent = send_store.for_subscription("SUB_A")
        assert len(sent) == 1
        record = sent[0]
        assert results[0] == results[1] == record
        # 金额与入算事实来自同一个数据版本：98 = 8 + 90，不跨版本混用
        assert record.amount == 98.0
        assert record.data_version == 2
        assert sorted(f.fact_id for f in record.facts) == ["F_A_DRAFT", "F_A_PUB"]
        assert record.amount == sum(f.amount for f in record.facts)
        assert verify_send_record(record, ledger)

    def test_recovery_requeries_instead_of_serving_stale_cache(
        self, ledger, cache, send_store, service, scope_a
    ):
        fetch_dashboard_data(scope_a, ledger, cache, now=0)  # 旧缓存 8 元（版本 1）
        ledger.publish("F_A_DRAFT")  # 版本升为 2，旧缓存失效
        misses_before = cache.misses

        recovered = SubscriptionService(ledger, cache, send_store, ttl_ticks=TTL)
        record = recovered.run_subscription("SUB_A", "SUB_A#run1", scope_a, now=1)

        # 旧版本缓存不得直接使用，按确认的查询范围重算
        assert cache.misses == misses_before + 1
        assert record.amount == 98.0
        # 下一次甲部取数与发件记录一致（同一版本的事实清单）
        nxt = fetch_dashboard_data(scope_a, ledger, cache, now=2)
        assert nxt.amount == record.amount == 98.0
        assert [f.fact_id for f in nxt.facts] == [f.fact_id for f in record.facts]

    def test_restart_resume_does_not_mix_departments(
        self, ledger, cache, send_store, service, scope_a, scope_b
    ):
        # 重启前甲部已发一件
        before = service.run_subscription("SUB_A", "SUB_A#1", scope_a, now=0)
        # 重启后重新进入该部门的订阅记录续作
        resumed = SubscriptionService(ledger, cache, send_store, ttl_ticks=TTL)
        again = resumed.run_subscription("SUB_A", "SUB_A#1", scope_a, now=1)
        assert again == before  # 同一运行号幂等，不重复发
        # 甲部的订阅记录只含甲部事实
        for record in send_store.for_department("甲部"):
            assert all(f.department == "甲部" for f in record.facts)
        # 乙部订阅发件不受甲部记录影响
        record_b = resumed.run_subscription("SUB_B", "SUB_B#1", scope_b, now=2)
        assert record_b.amount == 20.0
        assert all(f.department == "乙部" for f in record_b.facts)
        assert len(send_store.all()) == 2


# ============================================================
# 订阅详情：核对缓存事实与发件金额
# ============================================================

class TestSubscriptionDetail:
    def test_detail_matches_dashboard_and_cache(
        self, ledger, cache, service, scope_a
    ):
        record = service.run_subscription("SUB_A", "SUB_A#1", scope_a, now=0)
        dashboard = fetch_dashboard_data(scope_a, ledger, cache, now=0)

        detail = service.subscription_detail("SUB_A", now=0)
        assert detail["subscription_id"] == "SUB_A"
        assert len(detail["records"]) == 1
        item = detail["records"][0]
        # 关联到实际发件的金额
        assert item["amount"] == record.amount == 8.0
        # 订阅详情与部门看板核对同名缓存返回的事实
        assert item["cache_matches_record"] is True
        assert item["cache_fact_ids"] == [f.fact_id for f in dashboard.facts]
        assert item["consistent"] is True
        # 当次查询条件随详情返回
        assert item["query_scope"]["department"] == "甲部"
        assert item["query_scope"]["publish_status"] == PUBLISHED

    def test_detail_scoped_to_own_department(
        self, ledger, cache, send_store, service, scope_a, scope_b
    ):
        service.run_subscription("SUB_A", "SUB_A#1", scope_a, now=0)
        service.run_subscription("SUB_B", "SUB_B#1", scope_b, now=1)
        detail_a = service.subscription_detail("SUB_A", now=2)
        detail_b = service.subscription_detail("SUB_B", now=2)
        assert [r["department"] for r in detail_a["records"]] == ["甲部"]
        assert [r["department"] for r in detail_b["records"]] == ["乙部"]
        assert detail_a["records"][0]["amount"] == 8.0
        assert detail_b["records"][0]["amount"] == 20.0


# ============================================================
# 跨版本混用防护
# ============================================================

class TestCrossVersionGuards:
    def _record(self, scope_a, amount, facts, version):
        return SendRecord(
            subscription_id="SUB_A",
            run_id="SUB_A#x",
            department="甲部",
            scope=scope_a,
            amount=amount,
            facts=tuple(facts),
            data_version=version,
            sent_tick=0,
        )

    def test_mixed_version_record_rejected(self, ledger, scope_a):
        ledger.publish("F_A_DRAFT")  # 版本 2：已发布 {F_A_PUB, F_A_DRAFT}
        pub, draft = ledger.facts("甲部")[1], ledger.facts("甲部")[0]
        # 金额取新版本 98，事实清单却只有旧版本的 8 元事实
        mixed = self._record(scope_a, 98.0, [pub], 2)
        assert not verify_send_record(mixed, ledger)
        # 金额取旧版本 8，事实清单却混入新版本事实
        mixed2 = self._record(scope_a, 8.0, [pub, draft], 1)
        assert not verify_send_record(mixed2, ledger)

    def test_draft_or_foreign_department_facts_rejected(self, ledger, scope_a):
        draft = WalletFact("F_A_DRAFT", "甲部", 90.0, DRAFT, 1)
        foreign = WalletFact("F_B_PUB", "乙部", 20.0, PUBLISHED, 1)
        assert not verify_send_record(self._record(scope_a, 98.0, [draft], 1), ledger)
        assert not verify_send_record(self._record(scope_a, 20.0, [foreign], 1), ledger)

    def test_consistent_record_accepted(self, ledger, scope_a):
        fact = WalletFact("F_A_PUB", "甲部", 8.0, PUBLISHED, 1)
        assert verify_send_record(self._record(scope_a, 8.0, [fact], 1), ledger)


# ============================================================
# 综合分析管道集成（流 11）
# ============================================================

WALLET_FACTS = [
    {"fact_id": "F_A_PUB", "department": "甲部", "amount": 8.0,
     "status": PUBLISHED, "version": 1},
    {"fact_id": "F_A_DRAFT", "department": "甲部", "amount": 90.0,
     "status": DRAFT, "version": 1},
    {"fact_id": "F_B_PUB", "department": "乙部", "amount": 20.0,
     "status": PUBLISHED, "version": 1},
]

SUBSCRIPTIONS = [
    {"subscription_id": "SUB_A", "query_name": "部门钱包汇总", "department": "甲部"},
    {"subscription_id": "SUB_B", "query_name": "部门钱包汇总", "department": "乙部"},
]


class TestPipelineIntegration:
    def test_flow11_sends_published_amounts_only(self):
        result = pipeline_dashboard_subscription(WALLET_FACTS, SUBSCRIPTIONS)
        assert result["pipeline"] == "dashboard_def -> query_cache -> dashboard_subscription"
        assert result["total_sent"] == 2
        amounts = {s["department"]: s["amount"] for s in result["sent"]}
        assert amounts["甲部"] == 8.0   # 不计草稿 90
        assert amounts["乙部"] == 20.0
        for item in result["sent"]:
            assert item["publish_status"] == PUBLISHED
            assert item["query_name"] == "部门钱包汇总"
            assert item["fact_ids"]  # 入算数据随发件记录返回

    def test_full_pipeline_includes_flow11(self):
        result = run_full_analysis_pipeline(
            metrics=[], dimensions=[], events=[], experiments=[],
            attribution_models=[], anomaly_rules=[], segments=[],
            ranking_boards=[], joins=[], report_templates=[],
            sample_data={
                "wallet_facts": WALLET_FACTS,
                "dashboard_subscriptions": SUBSCRIPTIONS,
            },
        )
        assert "flow11_subscription" in result
        assert result["flow11_subscription"]["total_sent"] == 2

    def test_flow11_is_deterministic(self):
        r1 = pipeline_dashboard_subscription(WALLET_FACTS, SUBSCRIPTIONS)
        r2 = pipeline_dashboard_subscription(WALLET_FACTS, SUBSCRIPTIONS)
        assert r1 == r2
