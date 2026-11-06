"""Explicit, token-protected command runner for deployment maintainers.

Ordinary app visits never run reproduction. The independent CLI is executed
only after an authenticated maintenance form and an explicit button click.
"""
from pathlib import Path
import hmac
import json
import os
import subprocess
import sys
import streamlit as st


def render(config):
    st.subheader('배포 환경 재현 검사 · 관리자')
    st.caption('이 화면은 일반 시연 기능이 아닙니다. 현재 서버에서 별도 검사 명령을 명시적으로 실행합니다.')
    try:
        expected = str(st.secrets['deployment_admin']['verification_token'])
    except Exception:
        st.info('관리자 검사 기능이 구성되지 않았습니다.')
        return
    if len(expected) < 32:
        st.error('관리자 검사 설정을 확인해 주세요.')
        return
    with st.form('deployment_admin_auth', clear_on_submit=True):
        supplied = st.text_input('관리자 검사 토큰', type='password', autocomplete='off')
        authenticate = st.form_submit_button('관리자 확인')
    if authenticate:
        st.session_state['deployment_admin_authenticated'] = hmac.compare_digest(supplied.encode(), expected.encode())
        if not st.session_state['deployment_admin_authenticated']:
            st.error('관리자 확인에 실패했습니다.')
    if not st.session_state.get('deployment_admin_authenticated'):
        return
    # Changing/removing the secret immediately invalidates this session as well.
    identity = __import__('hashlib').sha256(expected.encode()).hexdigest()
    previous = st.session_state.get('deployment_admin_token_identity')
    if previous is not None and previous != identity:
        st.session_state['deployment_admin_authenticated'] = False
        return
    st.session_state['deployment_admin_token_identity'] = identity
    if st.button('이 배포 환경 재현 검사 실행', key='run_deployment_verification', type='primary'):
        env = os.environ.copy()
        env.update({'LIFTPOINT_BUNDLE_PATH': str(config.archive_path), 'LIFTPOINT_BUNDLE_SHA256': config.bundle_sha256,
                    'LIFTPOINT_RECORD_DIR': str(config.record_dir), 'LIFTPOINT_CODE_COMMIT': config.code_commit})
        # No admin token in command arguments, environment or persisted output.
        env.pop('LIFTPOINT_ADMIN_VERIFY_TOKEN', None)
        with st.spinner('별도 검사 명령 실행 중…'):
            try:
                result = subprocess.run([sys.executable, str(Path(__file__).with_name('verify_reproduction.py')), '--model', 'all'],
                                        cwd=Path(__file__).parent, env=env, capture_output=True, text=True, timeout=300)
                parsed = []
                for line in result.stdout.splitlines():
                    try:
                        item = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if item.get('model') in ('DL', 'ML'):
                        parsed.append({key: item[key] for key in ('model', 'status', 'atol', 'rtol', 'max_probability_abs_error', 'selected_symmetric_difference', 'seconds') if key in item})
                st.session_state['deployment_verification_result'] = parsed
                records = []
                from artifacts import read_json, sha256
                for model in ('DL', 'ML'):
                    pointer = read_json(config.record_dir / (model + '.latest.json'))
                    record_path = config.record_dir / pointer['record_file']
                    if record_path.resolve().parent == config.record_dir.resolve() and sha256(record_path) == pointer['record_sha256']:
                        records.append(read_json(record_path))
                st.session_state['deployment_verification_records'] = records
                if result.returncode == 0 and len(parsed) == 2 and all(item['status'] == 'passed' for item in parsed):
                    st.success('현재 배포 환경의 두 모델 재현 검사가 통과했습니다.')
                else:
                    st.error('일부 배포 재현 검사가 통과하지 못했습니다. 실제 추론은 해당 모델의 일치 기록이 있어야 활성화됩니다.')
            except Exception:
                st.error('별도 검사 명령을 완료하지 못했습니다. 서버 로그나 비밀정보를 화면에 표시하지 않습니다.')
        # Never display captured stderr/stdout outside the allowlisted aggregates.
    if st.session_state.get('deployment_verification_result'):
        st.json(st.session_state['deployment_verification_result'])
    if st.session_state.get('deployment_verification_records'):
        with st.expander('같은 배포 환경 검사 기록 · Secrets 보존용'):
            st.caption('고객 행·개별 확률 없이 파일/코드 해시·환경·집계 비교 결과만 포함합니다. Secrets에 보존하더라도 현재 지문이 달라지면 활성화되지 않습니다.')
            st.code(json.dumps({'records': st.session_state['deployment_verification_records']}, ensure_ascii=False, indent=2), language='json')
