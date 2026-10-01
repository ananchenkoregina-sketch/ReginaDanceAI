import base64
import os
import subprocess
import tempfile
from pathlib import Path

import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="Regina Dance AI",
    page_icon="💃",
    layout="centered",
    initial_sidebar_state="collapsed",
)

SYSTEM_PROMPT = """
Ты — персональный AI-SMM-стратег и контент-продюсер Регины.

Контекст бренда:
- Регина — хореограф, тренер по strip plastic, frame up, high heels и преподаватель народных танцев.
- Главная коммерческая задача на первом этапе — привлекать женщин на индивидуальные занятия по strip plastic.
- Контент должен выглядеть женственно, эстетично, уверенно и профессионально, без дешёвой провокационности.
- Народная хореография и профессиональная деятельность преподавателя могут использоваться для формирования экспертности и личности бренда, но не должны визуально смешиваться с продажей strip plastic в одном ролике без причины.
- Не выдумывай факты, отзывы, результаты учеников или события, которых нет в исходных данных.
- Не обещай гарантированный рост подписчиков/заявок.

Когда получаешь видео и запрос пользователя:
1. Определи, что реально происходит в видео.
2. Найди наиболее сильные визуальные моменты.
3. Предложи, как использовать материал для Instagram Reels.
4. Дай конкретную рекомендацию по монтажу: начало, порядок кадров, примерную длительность, текст на экране.
5. Напиши 3 разных hook-варианта.
6. Напиши готовую подпись к Reels.
7. Напиши CTA.
8. Предложи серию из 3-5 Stories как продолжение.
9. Если исходный материал слабый для конкретной цели, честно скажи это и предложи, что доснять.
10. Пиши по-русски, естественно, современно и без шаблонного "экспертного" канцелярита.

Формат ответа:
## 1. Что я вижу
## 2. Вердикт по материалу
## 3. Идеи Reels
## 4. Лучший вариант монтажа
## 5. Текст на экране
## 6. Подпись
## 7. CTA
## 8. Stories
## 9. Что доснять
"""

def get_api_key():
    # Cloud version: Streamlit Secrets.
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        key = ""
    if not key:
        key = os.getenv("OPENAI_API_KEY", "")
    return key

def run_cmd(args):
    return subprocess.run(args, capture_output=True, text=True, check=True)

def get_duration(video_path: str):
    probe = run_cmd([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ])
    return float(probe.stdout.strip())

def extract_frames(video_path: str, out_dir: str, count: int = 8):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    duration = get_duration(video_path)

    if duration <= 1:
        times = [0]
    else:
        start = min(0.5, duration * 0.05)
        end = max(start, duration - min(0.5, duration * 0.05))
        if count == 1:
            times = [duration / 2]
        else:
            times = [
                start + (end - start) * i / (count - 1)
                for i in range(count)
            ]

    paths = []
    for i, t in enumerate(times):
        frame_path = out / f"frame_{i:02d}.jpg"
        run_cmd([
            "ffmpeg", "-y",
            "-ss", str(t),
            "-i", video_path,
            "-frames:v", "1",
            "-vf", "scale=720:-1",
            "-q:v", "3",
            str(frame_path),
        ])
        paths.append((frame_path, t))
    return paths, duration

def extract_audio(video_path: str, out_path: str):
    try:
        run_cmd([
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "mp3",
            "-b:a", "64k",
            out_path,
        ])
        return Path(out_path)
    except Exception:
        return None

def transcribe_audio(client, video_path: str):
    audio_path = Path(video_path).with_suffix(".mp3")
    audio_path = extract_audio(video_path, str(audio_path))
    if not audio_path or not audio_path.exists() or audio_path.stat().st_size < 1000:
        return ""
    try:
        with open(audio_path, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=audio_file,
                language="ru",
            )
        return getattr(result, "text", "") or ""
    except Exception:
        return ""

def image_to_data_url(path: Path):
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"

api_key = get_api_key()

st.title("💃 Regina Dance AI")
st.caption("Твой персональный AI-SMM-помощник для танцевального Instagram")

if not api_key:
    st.error(
        "Приложение ещё не подключено к OpenAI. "
        "На этапе настройки нужно добавить OPENAI_API_KEY в Secrets Streamlit."
    )
    st.info("Если ты ещё не дошла до настройки Secrets — это нормально. Следуй инструкции ниже.")
    st.stop()

client = OpenAI(api_key=api_key)

with st.expander("⚙️ Настройки анализа", expanded=False):
    model = st.selectbox(
        "AI-модель",
        ["gpt-5.6-luna", "gpt-5.6-sol"],
        index=0,
        help="Luna — экономичнее; Sol — более мощный вариант.",
    )
    frame_count = st.slider("Количество кадров", 4, 12, 8)

goal = st.selectbox(
    "🎯 Что хотим получить?",
    [
        "Привлечь девушек на индивидуальные занятия",
        "Набрать подписчиков",
        "Показать экспертность",
        "Прогреть аудиторию",
        "Продать конкретную услугу",
        "Красиво оформить танцевальный материал",
    ],
)

uploaded = st.file_uploader(
    "🎥 Загрузи видео с iPad",
    type=["mp4", "mov", "m4v", "avi"],
    help="Можно выбрать видео из Фото или приложения «Файлы».",
)

user_request = st.text_area(
    "💬 Что ты хочешь получить из этого видео?",
    placeholder=(
        "Например: Сделай из этого продающий Reels для индивидуальных "
        "занятий по strip plastic. Хочу привлечь женщин 20–40 лет."
    ),
    height=110,
)

if uploaded:
    st.video(uploaded)

analyze = st.button(
    "✨ ПРОАНАЛИЗИРОВАТЬ ВИДЕО",
    type="primary",
    use_container_width=True,
)

if analyze:
    if not uploaded:
        st.warning("Сначала загрузи видео.")
        st.stop()

    request_text = user_request.strip() or f"Проанализируй видео с учётом цели: {goal}."

    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / uploaded.name
        video_path.write_bytes(uploaded.getbuffer())
        frames_dir = Path(tmp) / "frames"

        with st.status("Анализирую видео…", expanded=True) as status:
            try:
                st.write("1/4 — выделяю ключевые кадры")
                frame_paths, duration = extract_frames(
                    str(video_path), str(frames_dir), frame_count
                )

                st.write(f"2/4 — видео длительностью {duration:.1f} сек.")

                st.write("3/4 — проверяю речь в видео")
                transcript = transcribe_audio(client, str(video_path))

                st.write("4/4 — создаю контент-стратегию")

                content = [{
                    "type": "input_text",
                    "text": (
                        f"Цель контента: {goal}\n\n"
                        f"Запрос Регины: {request_text}\n\n"
                        f"Длительность видео: {duration:.1f} секунд.\n\n"
                        f"Расшифровка речи (если была): "
                        f"{transcript or 'речи не обнаружено'}\n\n"
                        "Ниже идут последовательные кадры из видео. "
                        "Учитывай их порядок при анализе."
                    ),
                }]

                for frame_path, timestamp in frame_paths:
                    content.append({
                        "type": "input_text",
                        "text": f"Кадр на отметке примерно {timestamp:.1f} сек.",
                    })
                    content.append({
                        "type": "input_image",
                        "image_url": image_to_data_url(frame_path),
                        "detail": "low",
                    })

                response = client.responses.create(
                    model=model,
                    instructions=SYSTEM_PROMPT,
                    input=[{"role": "user", "content": content}],
                )

                answer = response.output_text
                status.update(label="Готово 💃", state="complete")

            except FileNotFoundError:
                status.update(label="Ошибка", state="error")
                st.error(
                    "На сервере не найден FFmpeg. Проверь, что в GitHub-проекте "
                    "есть файл packages.txt со строкой ffmpeg."
                )
                st.stop()
            except Exception as e:
                status.update(label="Ошибка", state="error")
                st.error(f"Не удалось завершить анализ: {e}")
                st.stop()

    st.markdown("---")
    st.markdown(answer)

    st.download_button(
        "📥 Скачать результат",
        data=answer,
        file_name="regina_dance_ai_result.txt",
        mime="text/plain",
        use_container_width=True,
    )
