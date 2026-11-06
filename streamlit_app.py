"""LiftPoint: a local, post-experiment demonstration of frozen policies.

The default replay path deliberately does not import any model loader.
"""
from __future__ import annotations

import html
import math
from pathlib import Path

import streamlit as st

from artifacts import (
    artifact_paths,
    inference_status,
    load_aggregates,
    load_config,
    load_replay,
    replay_fingerprint,
)
from selection import anonymous_ids, select_indices


APP_DIR = Path(__file__).resolve().parent
MODEL_LABELS = {
    "DL": "DL · MLP 기준형, seed42",
    "ML": "ML · LogisticRegression C=0.1",
}
REPLAY = "저장 예측 재생"
LIVE = "실제 모델 추론"
SELECTION = "문자 발송 대상 선정"
RESULTS = "확정된 최종 검증 결과"
N_CUSTOMERS = 40008
N_SELECTED = 12002

st.set_page_config(page_title="LiftPoint", page_icon="◈", layout="wide", initial_sidebar_state="collapsed")


def esc(value) -> str:
    return html.escape(str(value))


def digest(value) -> str:
    """Accept the backend's full fingerprint or its digest string."""
    return value["digest"] if isinstance(value, dict) else str(value)


@st.cache_data(show_spinner=False, max_entries=3)
def cached_replay(_config, content_digest: str):
    return load_replay(_config)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_inference(_config, model: str, content_digest: str):
    # Import only after a live-mode request and a current, model-specific PASS.
    from inference import run_inference

    return run_inference(_config, model)


def rows(value):
    """Normalize stored aggregate records without recomputing any score."""
    if hasattr(value, "to_dict"):
        return value.to_dict(orient="records")
    return value


def number(value, decimals=0, suffix="") -> str:
    try:
        numeric = float(value)
    except (ValueError, TypeError):
        return "확인 불가"
    if not math.isfinite(numeric):
        return "복원 불가"
    return f"{numeric:,.{decimals}f}{suffix}"


def cards(items):
    cols = st.columns(len(items), gap="medium")
    for col, (label, value, note) in zip(cols, items):
        with col:
            st.markdown(
                f'<div class="lp-card"><div class="lp-card-label">{esc(label)}</div>'
                f'<div class="lp-card-value{ " lp-card-compact" if len(str(value)) > 18 else "" }">{esc(value)}</div>'
                f'<div class="lp-card-note">{esc(note)}</div></div>',
                unsafe_allow_html=True,
            )


def table(headers, body, css=""):
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    data = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in body)
    st.markdown(f'<div class="lp-table-wrap {css}"><table class="lp-table"><thead><tr>{head}</tr></thead><tbody>{data}</tbody></table></div>', unsafe_allow_html=True)


def render_interval(point, lower, upper):
    """Draw only the already stored point and percentile interval, in %p."""
    axis_lower, axis_upper = min(-1.5, lower - .25), max(2.25, upper + .25)
    def x(value):
        return 90 + (value - axis_lower) / (axis_upper - axis_lower) * 820
    ticks = [-1, 0, 1, 2]
    tick_markup = "".join(f'<line x1="{x(t):.2f}" x2="{x(t):.2f}" y1="204" y2="211" stroke="#7B8799"/><text x="{x(t):.2f}" y="239" text-anchor="middle" font-size="19" fill="#334155">{t:+.0f}</text>' for t in ticks)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 270" role="img" aria-label="DL−ML 선정 이득 차이 {point:+.4f}%p, 95% 구간 {lower:+.4f}부터 {upper:+.4f}%p. 구간에 0 포함.">
    <rect x="0" y="0" width="1000" height="270" fill="#F7FAFC" rx="16"/>
    <line x1="{x(0):.2f}" x2="{x(0):.2f}" y1="36" y2="205" stroke="#64748B" stroke-width="2" stroke-dasharray="6 5"/>
    <text x="{x(0):.2f}" y="26" text-anchor="middle" font-size="17" fill="#475569">차이 없음 · 0</text>
    <line x1="{x(lower):.2f}" x2="{x(upper):.2f}" y1="117" y2="117" stroke="#0F766E" stroke-width="7" stroke-linecap="round"/>
    <line x1="{x(lower):.2f}" x2="{x(lower):.2f}" y1="101" y2="133" stroke="#0F766E" stroke-width="3"/>
    <line x1="{x(upper):.2f}" x2="{x(upper):.2f}" y1="101" y2="133" stroke="#0F766E" stroke-width="3"/>
    <circle cx="{x(point):.2f}" cy="117" r="11" fill="#0F172A" stroke="white" stroke-width="3"/>
    <text x="{x(point):.2f}" y="76" text-anchor="middle" font-size="30" font-weight="700" fill="#0F172A">{point:+.4f}%p</text>
    <text x="{x(lower):.2f}" y="168" text-anchor="middle" font-size="22" font-weight="700" fill="#115E59">{lower:+.4f}</text>
    <text x="{x(upper):.2f}" y="168" text-anchor="middle" font-size="22" font-weight="700" fill="#115E59">{upper:+.4f}</text>
    <line x1="90" x2="910" y1="205" y2="205" stroke="#CBD5E1"/>{tick_markup}
    <text x="953" y="239" text-anchor="end" font-size="17" fill="#475569">%p</text>
    </svg>'''
    st.markdown(f'<div class="lp-chart">{svg}</div>', unsafe_allow_html=True)


st.markdown("""
<style>
:root { color-scheme: light; }
.stApp { background: #F4F7FA; color: #142337; }
.block-container { max-width: 1280px; padding-top: 2.0rem; padding-bottom: 3.0rem; }
header[data-testid="stHeader"] { background: rgba(244,247,250,.92); }
h1,h2,h3 { color: #10233B; font-weight: 750 !important; letter-spacing: -.035em; }
h1 { font-size: 3.15rem !important; line-height: 1.12 !important; padding-bottom: .1rem !important; }
h2 { font-size: 1.65rem !important; margin-top: 1rem !important; }
h3 { font-size: 1.3rem !important; }
p,li,label { font-size: 1.04rem; line-height: 1.65; }
div[data-testid="stCaptionContainer"] p { color: #526176; font-size: .94rem; }
div[data-testid="stRadio"] label p { font-size: 1.02rem; font-weight: 500; }
button[kind="primary"] { background: #0F766E; border-color: #0F766E; min-height: 3rem; font-weight: 700; }
button[kind="primary"]:hover { background: #115E59; border-color: #115E59; }
.lp-wordmark { color:#0F766E; font-size:.85rem; letter-spacing:.14em; font-weight:700; margin-bottom:.35rem; }
.lp-subtitle { font-size:1.4rem; color:#334155; margin-bottom:1.1rem; }
.lp-badge { display:inline-block; background:#DFEFEC; color:#115E59; font-size:.9rem; font-weight:650; padding:.35rem .8rem; border-radius:30px; }
.lp-context { color:#43536B; font-size:1rem; border-left:3px solid #0F766E; padding-left:.85rem; margin:.8rem 0 1.4rem; }
.lp-card { background:white; border:1px solid #DCE4EE; border-radius:15px; padding:1.25rem 1.35rem; min-height:145px; margin-bottom:1rem; }
.lp-card-label { color:#526176; font-size:1rem; font-weight:500; }
.lp-card-value { color:#10233B; font-size:2.05rem; font-weight:750; line-height:1.4; margin:.15rem 0; letter-spacing:-.04em; }
.lp-card-compact { font-size:1.6rem; }
.lp-card-note { color:#64748B; font-size:.88rem; line-height:1.5; }
.lp-intro { background:white; border:1px solid #DCE4EE; border-radius:16px; padding:1.7rem 2rem; margin:1rem 0; }
.lp-intro h3 { margin:0 0 .8rem; }
.lp-intro p { color:#43536B; margin:.35rem 0; }
.lp-table-wrap { overflow-x:auto; border:1px solid #DCE4EE; border-radius:14px; background:white; margin:.5rem 0 1rem; }
.lp-table { width:100%; border-collapse:collapse; font-size:1rem; text-align:left; }
.lp-table th { padding:1rem 1.1rem; background:#EAF0F5; color:#334155; font-size:.94rem; font-weight:650; white-space:nowrap; }
.lp-table td { padding:.75rem 1.1rem; border-top:1px solid #E5EBF1; color:#23364F; font-variant-numeric:tabular-nums; white-space:nowrap; }
.lp-table td:not(:first-child), .lp-table th:not(:first-child) { text-align:right; }
.lp-table tr:last-child td { border-bottom:0; }
.lp-outcome { font-size:1.18rem; font-weight:600; color:#10233B; background:#E5EFED; border-left:4px solid #0F766E; border-radius:8px; padding:1.1rem 1.3rem; margin:1rem 0 1.4rem; line-height:1.65; }
.lp-chart { background:#F7FAFC; border:1px solid #DCE4EE; border-radius:16px; padding:.5rem; }
.lp-chart svg { display:block; width:100%; height:auto; }
.lp-rule { color:#526176; font-size:.94rem; margin:.5rem 0 1.1rem; }
.lp-footer { color:#64748B; border-top:1px solid #DCE4EE; margin-top:2rem; padding-top:1rem; font-size:.9rem; }
[data-testid="stToolbar"] { visibility:hidden; }
@media(max-width:760px) { .block-container {padding-top:1.1rem;} h1 {font-size:2.5rem !important;} .lp-card-value {font-size:1.6rem;} .lp-table td,.lp-table th {padding:.65rem;} }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="lp-wordmark">SMS TARGETING DEMO</div>', unsafe_allow_html=True)
st.title("LiftPoint")
st.markdown('<div class="lp-subtitle">문자 발송 대상 선정 시연</div>', unsafe_allow_html=True)

mode = st.radio("운영 방식", [REPLAY, LIVE], horizontal=True, key="mode")
st.markdown(f'<span class="lp-badge">{esc(mode)}</span><div class="lp-context">최종 테스트 고객 <b>40,008명의 사후 시연</b> · 고정된 모델과 기존 실험 산출물을 사용합니다.</div>', unsafe_allow_html=True)
screen = st.radio("화면", [SELECTION, RESULTS], horizontal=True, key="screen", label_visibility="collapsed")

try:
    config = load_config()
except Exception as error:
    st.error(f"LiftPoint 경로 설정을 확인해 주세요. ({type(error).__name__})")
    st.caption("README의 경로 설정 방법을 확인한 뒤 앱을 다시 실행해 주세요.")
    st.stop()


if st.query_params.get('admin') == 'verify':
    from admin_verify import render
    render(config)
    st.stop()

if screen == RESULTS:
    st.header("확정된 최종 검증 결과")
    st.caption("이미 완료한 고정 최종 테스트의 집계 결과입니다. 현재 모드와 무관하게 저장된 수치만 표시합니다.")
    try:
        aggregates = load_aggregates(config)
        scorecard = {row["policy"]: row for row in rows(aggregates["scorecard"])}
        interval = next(row for row in rows(aggregates["intervals"]) if row["metric"] == "DL_gain_minus_ML_gain")
        point, lower, upper = (float(interval[key]) for key in ("point_pp", "lower_95_pp", "upper_95_pp"))
        cards([
            ("DL−ML 선정 이득 차이", f"{point:+.4f}%p", "점추정 · 두 고정 정책의 차이"),
            ("95% 구간", f"{lower:+.4f} ~ {upper:+.4f}", "%p · 0 포함"),
            ("기존 paired bootstrap", "2,000회", "seed 20261105 · 95% percentile"),
        ])
        st.subheader("DL−ML 선정 이득 차이")
        render_interval(point, lower, upper)
        st.markdown('<div class="lp-outcome">DL의 점추정은 높았지만, 차이의 95% 구간에 0이 포함돼 확실한 우위는 확인하지 못했습니다.</div>', unsafe_allow_html=True)
        st.subheader("두 대표 모델의 확정 집계")
        table(
            ["모델", "선정군 효과", "무작위 기대 기준 대비 이득", "관측 BCE"],
            [[MODEL_LABELS[model], f"{float(scorecard[model]['selected_effect_pp']):.4f}%p", f"{float(scorecard[model]['selection_gain_pp']):.4f}%p", f"{float(scorecard[model]['factual_bce']):.6f}"] for model in ("DL", "ML")],
        )
        st.markdown("**무작위 선정 기대 기준**")
        st.write("전체 테스트 고객의 발송군 구매율과 미발송군 구매율 차이로 추정한 무작위 선정 기대 기준입니다.")
        st.caption("효과 수치는 발송 배정의 비교가능성 등 가정 아래의 추정입니다. 구간은 고정 정책의 평가 고객 표본 불확실성을 나타내며, 학습·후보 선택·배정 편향 전체를 포함하지 않습니다.")
    except Exception as error:
        st.error(f"확정 집계 파일을 확인할 수 없습니다. ({type(error).__name__})")
else:
    st.header("문자 발송 대상 선정")
    model = st.selectbox("대표 모델", ["DL", "ML"], format_func=MODEL_LABELS.get, key="model")
    status = None
    available = True
    try:
        replay_version = replay_fingerprint(config)
        replay_key = digest(replay_version)
    except Exception as error:
        st.error(f"저장 예측 재생에 필요한 파일을 확인할 수 없습니다. ({type(error).__name__})")
        st.stop()
    current_key = (model, mode, replay_key)
    if mode == LIVE:
        status = inference_status(config, model)
        available = bool(status.get("enabled"))
        if available:
            current_key = (model, mode, replay_key, digest(status["fingerprint"]))
            st.caption("현재 파일의 내용 해시와 별도 재현 검사 통과 기록이 일치합니다. 버튼을 누르면 고정 모델로 추론합니다.")
        else:
            st.warning("이 모델의 실제 추론이 비활성화되어 있습니다. 저장 예측 재생을 선택하면 시연을 계속할 수 있습니다.")
            st.caption(str(status.get("reason", "현재 산출물과 일치하는 재현 검사 통과 기록이 필요합니다.")))
    else:
        st.caption("기존 최종 예측을 읽습니다. 모델 로더를 불러오지 않습니다.")
    cards([
        ("시연 대상", "40,008명", "고정 최종 테스트 고객"),
        ("문자 발송 선정", "12,002명", "상위 30% 고정 정책"),
        ("선정 점수", "p1 − p0", "미반올림 값 내림차순"),
    ])
    st.markdown('<div class="lp-rule">동점에는 기존 고정 해시 규칙을 적용합니다. 실제 발송 여부와 구매 정답은 모델 입력·정렬·선정에 사용하지 않습니다.</div>', unsafe_allow_html=True)
    clicked = st.button("선정 결과 보기", type="primary", disabled=not available, key="show_selection")
    if clicked:
        try:
            with st.spinner("선정 결과를 준비하고 있습니다…"):
                replay = cached_replay(config, replay_key)
                probabilities = replay.predictions[model] if mode == REPLAY else cached_inference(config, model, digest(status["fingerprint"]))
                order = select_indices(replay.ids, probabilities)
                if len(replay.ids) != N_CUSTOMERS or len(order) != N_CUSTOMERS:
                    raise ValueError("Unexpected customer count")
                st.session_state["selection_result"] = {"key": current_key, "probabilities": probabilities, "order": order}
        except Exception as error:
            st.session_state.pop("selection_result", None)
            st.error(f"선정 결과를 준비하지 못했습니다. ({type(error).__name__})")
            if mode == LIVE:
                st.info("저장 예측 재생으로 전환하면 기존 예측으로 시연할 수 있습니다. 실제 추론 오류는 별도 재현 검사 명령으로 확인해 주세요.")
    saved = st.session_state.get("selection_result")
    if saved and tuple(saved["key"]) == current_key and available:
        replay = cached_replay(config, replay_key)
        probabilities = saved["probabilities"]
        order = saved["order"]
        aliases = anonymous_ids(replay.ids)
        selected_set = set(int(i) for i in order[:N_SELECTED])
        rank_by_index = {int(index): rank + 1 for rank, index in enumerate(order)}
        st.subheader("선정 고객 확인")
        st.write("**정확히 12,002명 선정** · 처음에는 선정 점수 순으로 20명만 표시합니다.")
        query = st.text_input("익명 고객 ID 검색", placeholder="익명 ID의 일부를 입력하세요", key="customer_search").strip().upper()
        candidates = [int(i) for i in order if query in str(aliases[int(i)]).upper()] if query else [int(i) for i in order[:N_SELECTED]]
        preview = candidates[:20]
        if query:
            st.caption(f"검색 결과 {len(candidates):,}명 · 최대 20명 표시")
        table(
            ["순위", "익명 고객 ID", "p0", "p1", "예상 증가분", "선정 여부"],
            [[f"{rank_by_index[i]:,}", aliases[i], f"{float(probabilities[i, 0]) * 100:.2f}%", f"{float(probabilities[i, 1]) * 100:.2f}%", f"{float(probabilities[i, 1] - probabilities[i, 0]) * 100:+.2f}%p", "선정" if i in selected_set else "미선정"] for i in preview],
        )
        if preview:
            chosen = st.selectbox("고객 상세", preview, format_func=lambda i: str(aliases[i]), key=f"detail_{model}_{mode}_{query}")
            st.subheader(f"고객 상세 · {aliases[chosen]}")
            st.caption("선정" if chosen in selected_set else "미선정")
            cards([
                ("p0 · 미발송 시 구매확률", f"{float(probabilities[chosen, 0]) * 100:.2f}%", "모델 추정"),
                ("p1 · 발송 시 구매확률", f"{float(probabilities[chosen, 1]) * 100:.2f}%", "모델 추정"),
                ("예상 증가분", f"{float(probabilities[chosen, 1] - probabilities[chosen, 0]) * 100:+.2f}%p", "p1 − p0"),
            ])
            summaries = replay.summaries
            def summary(key):
                values = summaries.get(key)
                return values[chosen] if values is not None else None
            table(
                ["관측 구매 요약", "원래 단위 값"],
                [["전체 관측 구매 횟수", number(summary("receipt_count"), suffix="회")],
                 ["전체 관측 총금액", number(summary("amount_total"), decimals=2)],
                 ["마지막 관측 구매 이후 경과일", number(summary("recency_days"), decimals=2, suffix="일")],
                 ["최근 30일 구매 횟수", number(summary("recent30_receipts"), suffix="회")],
                 ["이전 30일 구매 횟수", number(summary("previous30_receipts"), suffix="회")]],
            )
            st.caption("경과일의 고정 기준 시각: 2019-03-19 00:00:00 · 금액은 원자료 단위입니다. 구매 1회는 영수증 1건입니다.")
            if chosen in selected_set:
                st.info("모델이 추정한 구매확률 증가분이 높아 선정된 고객입니다. 개인별 실제 문자 효과를 관측한 값은 아닙니다.")
            else:
                st.info("고정 정책의 상위 12,002명에 포함되지 않은 고객입니다. 개인별 실제 문자 효과를 관측한 값은 아닙니다.")
        else:
            st.info("일치하는 익명 고객 ID가 없습니다.")
    else:
        st.markdown('<div class="lp-intro"><h3>같은 발송 규모에서, 누구를 선정할까요?</h3><p>두 대표 모델이 추정한 구매확률 증가분으로 고객의 순서를 정합니다.</p><p><b>01</b> 대표 모델 선택　→　<b>02</b> 선정 결과 보기　→　<b>03</b> 고객의 p0·p1·증가분 확인</p><p>다음 화면에서 확정된 최종 결과와 차이의 불확실성을 확인할 수 있습니다.</p></div>', unsafe_allow_html=True)

with st.expander("실행 정보"):
    st.write("Incremental Value 실험 결과를 활용한 LiftPoint 시연 앱")
    st.caption("고정 대표: DL MLP 기준형 seed42 · ML LogisticRegression C=0.1, 기존 p0·p1 모델")
    st.write("확정 실험 근거 SHA: `e1d52d29a7ae9145a7637b14a0c3aefc14602b56`")
    try:
        evidence = artifact_paths(config)
        st.write(f"입력 버전: `{evidence['freeze']['preprocessing']['input_version']}` · 고객 40,008명 × 변수 91개")
        st.write(f"저장 예측 버전: `{evidence['output_dir'].name}/final_predictions.npz`")
        st.write(f"저장 예측 SHA-256: `{evidence['bundle']['files']['predictions/final_predictions.npz']['sha256']}`")
        st.write(f"배포 코드 커밋: `{config.code_commit}`")
        st.write(f"freeze SHA-256: `{evidence['input_meta']['freeze_sha256']}`")
        info = replay_fingerprint(config)
        st.write("입력·저장 예측·전처리·선정 코드의 현재 버전")
        if isinstance(info, dict):
            st.json(info, expanded=False)
        else:
            st.code(str(info), language=None)
        for kind in ("DL", "ML"):
            item = inference_status(config, kind)
            st.write(f"**{MODEL_LABELS[kind]} 실제 추론:** {'활성화' if item.get('enabled') else '비활성화'}")
            st.caption(str(item.get("reason", "")))
            if item.get("fingerprint"):
                st.json(item["fingerprint"], expanded=False)
    except Exception as error:
        st.caption(f"일부 실행 정보 확인 불가: {type(error).__name__}")
    st.caption("현재 산출물과 일치하는 모델별 재현 검사 통과 기록이 있어야 실제 추론이 활성화됩니다. 앱에서는 전체 재현 검사를 실행하지 않습니다.")

st.markdown('<div class="lp-footer">LiftPoint · 확정 실험 결과를 활용한 시연 앱 · 최종 테스트 40,008명 / 선정 12,002명</div>', unsafe_allow_html=True)
