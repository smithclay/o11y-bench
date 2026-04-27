# O11y Skill

Solve Grafana observability tasks (Prometheus, Loki, Tempo, dashboards) by
composing a small set of high-level Python helpers in the REPL.

## Workflow

1. The user message includes a `<context>` block with the synthetic `Current
   time:`. Treat that as "now" and derive explicit ISO-8601 `start`/`end`
   bounds from it for time-windowed queries.
2. Identify the single signal that answers the question before issuing
   queries. Resist the urge to gather breadth — every extra query bloats your
   context and rarely helps the answer.
3. Execute exactly the queries you need, in parallel via `asyncio.gather`
   when comparing N candidates.
4. Submit a concise final answer in `answer` — no preamble, no hedging, no
   restated reasoning. The grader scores the literal text against a rubric.

## Tools

All tools are async — `await` every call. Datasource UIDs are resolved
internally; you never need to pass them.

- `await list_datasources()` — returns `[{uid, name, type}]` for the stack.
  Useful for discovery; rarely needed once helpers are in use.
- `await query_metrics(expr, start=None, end=None, step="30s")` — PromQL.
  Range query if both `start` and `end` are given (ISO-8601), instant query
  otherwise.
- `await query_logs(expr, start, end, limit=100)` — LogQL.
- `await query_traces(traceql, start=None, end=None, limit=20)` — TraceQL.
- `await get_dashboard(uid)` / `await save_dashboard(model)` /
  `await search_dashboards(query)` — dashboard CRUD.

Numeric claims must come from query results, not LM intuition. The grader
re-runs reference queries; mismatches lose points.

## PromQL idioms

- Counters end in `_total`. Always wrap them in `rate()` (per-second over a
  range) or `increase()` (raw delta over a range). A bare counter value is
  cumulative since process start and almost never the right answer.
- Range vector vs instant: `metric` is instant, `metric[5m]` is a range
  vector. Range vectors only feed range-aware functions (`rate`, `increase`,
  `avg_over_time`, ...).
- For "over the last X" phrasing, run a range query (with `start`/`end`/`step`)
  and aggregate over the window — do not run an instant query against `now`.
- `topk(N, expr)` for ranking, `bottomk(N, expr)` for the inverse. Aggregate
  with `sum by (label)(...)` before ranking.
- Percentiles:
  `histogram_quantile(0.95, sum by (le, ...) (rate(metric_bucket[5m])))`.
  Always aggregate by `le` plus the labels you want to keep.
- `label_replace` cannot create labels that did not exist; it copies/edits.
  When joining series with `*`/`/`, use `on()`/`ignoring()`/`group_left`.

## LogQL idioms

- Always start with the smallest viable label matcher: `{job="x", level="error"}`
  first, line filter `|= "foo"` second, parser `| json` / `| logfmt` last.
  Cardinality matters.
- Peek at a few raw lines before parsing (`limit=5`, no parser) so you know the
  actual log shape before you commit to extraction code.
- Aggregations: `sum by (label) (count_over_time({...}[5m]))` for counts,
  `sum by (label) (rate({...}[5m]))` for per-second rates.
- For numeric extraction, `| json | unwrap field | rate(...)` works on numeric
  fields; `| logfmt` for `key=value` logs.
- `limit` only caps returned lines, not the underlying scan; keep `start`/`end`
  ranges tight.

## TraceQL idioms

- Filter on `resource.service.name="x"` and `span.status=error` before any
  duration filter. Tempo can short-circuit on these.
- Spec phrasing: `{ resource.service.name = "checkout" && span.status = error }`.
- Project explicitly with `| select(span.name, span.http.status_code,
  resource.service.name, duration)` — rarely want the full span.
- For latency: `{ resource.service.name = "x" } | select(duration) |
  histogram_over_time(duration)` and inspect the shape, not raw samples.
- A `trace_id` from one query can be reused in `{ trace:id = "..." }` to pull
  the full trace once you know which one matters.

## Dashboard editing

- Round-trip the whole panel JSON: `await get_dashboard(uid)` → mutate →
  `await save_dashboard(model)`. Never construct a partial dashboard.
- After saving, ALWAYS re-fetch with `get_dashboard(uid)` and assert the saved
  expression / panel title / unit / variable binding actually matches what you
  intended. Saves can fail silently if the schema is off.
- Variables live at `dashboard.templating.list[*]`. A panel's
  `targets[*].expr` must reference variables that exist there.
- Dashboard uids and folder uids share a namespace. Don't collide them.

## Never fabricate

The grader re-runs the canonical query and compares your numeric answer.
A made-up plausible-looking number scores **zero**, the same as no answer.

- **Never** use `await predict(...)` to invent or summarize *numeric* values
  that should come from a query. `predict()` is for structured *extraction*
  from real query results, not for generating answers when queries fail.
- If queries fail or return empty data after a real attempt, say so
  explicitly in the answer (e.g. *"queried `process_cpu_seconds_total` over
  the 6h window; result was empty — no CPU data available for that
  metric/window"*) instead of inventing values.
- If a tool helper raises an unexpected error, surface the error message in
  your answer so the user/grader can see what actually happened, instead of
  paraphrasing the failure as a high-level "infrastructure issue".

## Answer formatting

- Concise. The grader rubric scores literal text against weighted criteria.
- For ranking questions, list ranks explicitly (e.g. `1. service-a (0.42 cores
  avg)`). Use the units the user asked for.
- For "what's the cause" questions, distinguish between *strong* and *weak*
  signals — the rubric usually rewards interpretation, not just fact.
- For numeric answers, prefer ~2 significant figures unless the task asks for
  more. Don't over-precise outputs the grader's fact-check just compares
  approximately.
