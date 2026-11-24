"""洞察工坊 · 综合分析看板订阅

根因：看板取数入口按发布状态过滤指标数据，订阅发件入口却用了另一份
未过滤草稿的口径，导致订阅结果错误计入未发布金额；同名查询的缓存键
又只看查询名称，两个部门的同名查询互相命中对方缓存。

修复：
- 看板展示与订阅发件共用同一个"确认的查询范围"取数入口，发布范围
  （已发布）只有一份定义，订阅结果不再计入草稿金额。
- 缓存键由完整查询条件（查询名称 + 部门 + 发布范围 + 时间范围）派生，
  同名查询跨部门互不命中；过期或数据版本失效即按原部门范围重查。
- 发件记录携带当次查询条件和入算数据，按订阅运行号幂等落库：
  重启恢复、并发恢复同一次订阅只发一份完整查询结果，
  发件记录与已发布数据不跨版本混用。
"""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from typing import Any, Callable

PUBLISHED = "已发布"
DRAFT = "草稿"


# ============================================================
# 部门钱包事实源（持久层：重启后事实与版本仍在）
# ============================================================

@dataclass(frozen=True)
class WalletFact:
    """一条部门钱包事实：金额、发布状态、所属数据版本。"""
    fact_id: str
    department: str
    amount: float
    status: str
    version: int


class DepartmentWalletLedger:
    """部门钱包台账：事实按部门归档，发布动作抬升部门数据版本。

    台账是持久层：服务重启后重新进入同一部门，看到的仍是这份事实，
    续作时不会混入别的部门数据。
    """

    def __init__(self, facts: list[WalletFact] | None = None) -> None:
        self._facts: dict[str, WalletFact] = {f.fact_id: f for f in facts or []}
        self._versions: dict[str, int] = {}
        # 每个数据版本对应的已发布事实快照，用于校验发件记录不跨版本混用
        self._history: dict[str, dict[int, frozenset[str]]] = {}
        for dept in {f.department for f in self._facts.values()}:
            version = max(
                (f.version for f in self._facts.values() if f.department == dept),
                default=0,
            )
            self._versions[dept] = version
            self._history.setdefault(dept, {})[version] = self._published_ids(dept)
        self._lock = threading.Lock()

    def _published_ids(self, department: str) -> frozenset[str]:
        return frozenset(
            f.fact_id
            for f in self._facts.values()
            if f.department == department and f.status == PUBLISHED
        )

    def current_version(self, department: str) -> int:
        return self._versions.get(department, 0)

    def facts(self, department: str) -> list[WalletFact]:
        return sorted(
            (f for f in self._facts.values() if f.department == department),
            key=lambda f: f.fact_id,
        )

    def published_facts(self, department: str) -> list[WalletFact]:
        """看板口径：只取已发布事实，草稿不计入。"""
        return [f for f in self.facts(department) if f.status == PUBLISHED]

    def published_ids_at(self, department: str, version: int) -> frozenset[str] | None:
        """指定数据版本下已发布事实的快照；未知版本返回 None。"""
        return self._history.get(department, {}).get(version)

    def publish(self, fact_id: str) -> WalletFact:
        """草稿发布：状态流转为已发布，并抬升部门数据版本。"""
        with self._lock:
            fact = self._facts[fact_id]
            if fact.status == PUBLISHED:
                return fact
            new_version = self.current_version(fact.department) + 1
            updated = WalletFact(
                fact.fact_id, fact.department, fact.amount, PUBLISHED, new_version
            )
            self._facts[fact_id] = updated
            self._versions[fact.department] = new_version
            self._history.setdefault(fact.department, {})[new_version] = (
                self._published_ids(fact.department)
            )
            return updated


# ============================================================
# 确认的查询范围
# ============================================================

@dataclass(frozen=True)
class ConfirmedQueryScope:
    """当次确认的查询条件：看板取数与订阅发件共用同一份范围。

    发布范围（publish_status）默认且只能取"已发布"，从结构上保证
    订阅发件入口与看板取数入口的发布口径一致。
    """
    query_name: str
    department: str
    publish_status: str = PUBLISHED
    time_range: str = ""

    def cache_key(self) -> str:
        """缓存键由完整查询条件派生：同名查询跨部门、跨口径互不命中。"""
        parts = [self.query_name, self.department, self.publish_status, self.time_range]
        return "scope:" + "|".join(parts)


@dataclass(frozen=True)
class QueryResult:
    """一次取数结果：指标金额 + 入算事实清单 + 数据版本。"""
    scope: ConfirmedQueryScope
    amount: float
    facts: tuple[WalletFact, ...]
    data_version: int
    from_cache: bool


def execute_confirmed_query(
    scope: ConfirmedQueryScope,
    ledger: DepartmentWalletLedger,
) -> QueryResult:
    """按确认的查询范围执行取数：只计入该部门已发布事实。"""
    facts = tuple(
        f for f in ledger.published_facts(scope.department)
        if f.status == scope.publish_status
    )
    return QueryResult(
        scope=scope,
        amount=sum(f.amount for f in facts),
        facts=facts,
        data_version=ledger.current_version(scope.department),
        from_cache=False,
    )


# ============================================================
# 看板查询缓存
# ============================================================

@dataclass
class CacheEntry:
    key: str
    result: QueryResult
    created_tick: int
    ttl_ticks: int

    def expired(self, now: int) -> bool:
        return now - self.created_tick >= self.ttl_ticks


class DashboardQueryCache:
    """看板查询缓存：按完整查询条件键控。

    过期或部门数据版本失效的条目立即淘汰，下一次取数按原范围重查，
    不会把旧版本或别的部门的结果返回给调用方。
    """

    def __init__(self) -> None:
        self._entries: dict[str, CacheEntry] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str, now: int, current_version: int) -> QueryResult | None:
        entry = self._entries.get(key)
        if entry is None or entry.expired(now) or entry.result.data_version != current_version:
            if entry is not None:
                del self._entries[key]
            self.misses += 1
            return None
        self.hits += 1
        r = entry.result
        return QueryResult(r.scope, r.amount, r.facts, r.data_version, from_cache=True)

    def peek(self, key: str, now: int, current_version: int) -> QueryResult | None:
        """只读查看缓存（订阅详情核对用），不淘汰条目、不计命中。"""
        entry = self._entries.get(key)
        if entry is None or entry.expired(now) or entry.result.data_version != current_version:
            return None
        return entry.result

    def put(self, key: str, result: QueryResult, now: int, ttl_ticks: int) -> None:
        self._entries[key] = CacheEntry(key, result, now, ttl_ticks)

    def keys(self) -> list[str]:
        return sorted(self._entries)


def fetch_dashboard_data(
    scope: ConfirmedQueryScope,
    ledger: DepartmentWalletLedger,
    cache: DashboardQueryCache,
    now: int,
    ttl_ticks: int = 300,
) -> QueryResult:
    """看板取数唯一入口：看板展示与订阅发件都从这里取数。

    先按完整查询条件查缓存；未命中、过期或版本失效时按确认的
    查询范围重算并回填，保证两个入口的发布范围始终一致。
    """
    key = scope.cache_key()
    cached = cache.get(key, now, ledger.current_version(scope.department))
    if cached is not None:
        return cached
    result = execute_confirmed_query(scope, ledger)
    cache.put(key, result, now, ttl_ticks)
    return result


# ============================================================
# 订阅发件记录（持久层）
# ============================================================

@dataclass(frozen=True)
class SendRecord:
    """看板订阅发件记录：当次查询条件 + 入算数据 + 实际发件金额。"""
    subscription_id: str
    run_id: str
    department: str
    scope: ConfirmedQueryScope
    amount: float
    facts: tuple[WalletFact, ...]
    data_version: int
    sent_tick: int


class SendRecordStore:
    """发件记录存储：按订阅运行号幂等，重启后记录仍在。

    同一次订阅运行（run_id）无论恢复多少次、多少方并发恢复，
    都只落一份完整的发件记录。
    """

    def __init__(self) -> None:
        self._records: dict[str, SendRecord] = {}
        self._lock = threading.Lock()

    def get(self, run_id: str) -> SendRecord | None:
        return self._records.get(run_id)

    def send_once(self, run_id: str, producer: Callable[[], SendRecord]) -> SendRecord:
        """同一次订阅运行只发一份：已发件的直接返回原记录，不重复发。"""
        with self._lock:
            existing = self._records.get(run_id)
            if existing is not None:
                return existing
            record = producer()
            self._records[run_id] = record
            return record

    def for_subscription(self, subscription_id: str) -> list[SendRecord]:
        return [r for r in self._records.values() if r.subscription_id == subscription_id]

    def for_department(self, department: str) -> list[SendRecord]:
        return [r for r in self._records.values() if r.department == department]

    def all(self) -> list[SendRecord]:
        return list(self._records.values())


def verify_send_record(record: SendRecord, ledger: DepartmentWalletLedger) -> bool:
    """校验发件记录不跨版本混用、不混入他部门数据、不计入草稿。

    - 金额必须等于入算事实之和；
    - 入算事实全部属于记录所属部门且全部为已发布；
    - 入算事实集合必须等于该数据版本下已发布事实的完整快照，
      不允许一半取旧版本、一半取新版本。
    """
    if record.scope.department != record.department:
        return False
    if record.scope.publish_status != PUBLISHED:
        return False
    if record.amount != sum(f.amount for f in record.facts):
        return False
    if any(f.department != record.department for f in record.facts):
        return False
    if any(f.status != PUBLISHED for f in record.facts):
        return False
    snapshot = ledger.published_ids_at(record.department, record.data_version)
    if snapshot is not None and frozenset(f.fact_id for f in record.facts) != snapshot:
        return False
    return True


# ============================================================
# 看板订阅服务
# ============================================================

class SubscriptionService:
    """看板订阅发件：与看板共用取数入口，重启恢复 exactly-once。

    服务实例本身是内存态（重启即换新的实例），事实台账、查询缓存、
    发件记录存储是持久层，重启后重新进入同一部门的订阅记录续作。
    """

    def __init__(
        self,
        ledger: DepartmentWalletLedger,
        cache: DashboardQueryCache,
        send_store: SendRecordStore,
        ttl_ticks: int = 300,
    ) -> None:
        self._ledger = ledger
        self._cache = cache
        self._send_store = send_store
        self._ttl_ticks = ttl_ticks

    def run_subscription(
        self,
        subscription_id: str,
        run_id: str,
        scope: ConfirmedQueryScope,
        now: int,
    ) -> SendRecord:
        """执行一次订阅发件。

        取数走看板同一个入口（确认的查询范围），发件记录按 run_id
        幂等落库：重启恢复或并发恢复同一次订阅，只发一份完整查询
        结果，且金额与入算事实来自同一个数据版本。
        """
        def produce() -> SendRecord:
            result = fetch_dashboard_data(
                scope, self._ledger, self._cache, now, self._ttl_ticks
            )
            return SendRecord(
                subscription_id=subscription_id,
                run_id=run_id,
                department=scope.department,
                scope=scope,
                amount=result.amount,
                facts=result.facts,
                data_version=result.data_version,
                sent_tick=now,
            )

        return self._send_store.send_once(run_id, produce)

    def subscription_detail(self, subscription_id: str, now: int) -> dict[str, Any]:
        """订阅详情：实际发件金额、入算事实清单，并核对同名缓存返回的事实。"""
        records = sorted(
            self._send_store.for_subscription(subscription_id),
            key=lambda r: (r.sent_tick, r.run_id),
        )
        items = []
        for record in records:
            cache_view = self._cache.peek(
                record.scope.cache_key(),
                now,
                self._ledger.current_version(record.department),
            )
            cache_fact_ids = (
                sorted(f.fact_id for f in cache_view.facts)
                if cache_view is not None else None
            )
            record_fact_ids = sorted(f.fact_id for f in record.facts)
            items.append({
                "run_id": record.run_id,
                "department": record.department,
                "query_scope": {
                    "query_name": record.scope.query_name,
                    "department": record.scope.department,
                    "publish_status": record.scope.publish_status,
                    "time_range": record.scope.time_range,
                },
                "amount": record.amount,
                "facts": [asdict(f) for f in record.facts],
                "data_version": record.data_version,
                "cache_fact_ids": cache_fact_ids,
                "cache_matches_record": (
                    cache_fact_ids == record_fact_ids
                    if cache_fact_ids is not None else None
                ),
                "consistent": verify_send_record(record, self._ledger),
            })
        return {"subscription_id": subscription_id, "records": items}
