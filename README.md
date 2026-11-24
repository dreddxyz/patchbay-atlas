# Patchbay Atlas

[Лаборатория поиска](https://dreddxyz.github.io/patchbay-atlas/). По формулировке проблемы со звуком найдите фрагмент студийного справочника и проверьте, на какие слова опирался ответ модели.

В репозитории 12 оригинальных англоязычных заметок и 24 размеченных вопроса, по два на заметку. Корпус и вопросы опубликованы отдельно. Это небольшой авторский benchmark; возможен уклон в формулировках. Результаты не показывают качество на реальных обращениях пользователей.

## Как устроен эксперимент

Baseline — TF–IDF с униграммами и биграммами, обученный только на корпусе. Для семантического поиска используются предобученный Hugging Face MiniLM, masked mean pooling и L2-нормализация в PyTorch. Сборка экспортирует encoder в ONNX и сравнивает эмбеддинги ONNX Runtime и PyTorch на всех заметках и вопросах. Qdrant Local хранит эмбеддинги и ищет по косинусной близости. Top-1, Recall@3 и MRR@3 считаются по эталонной заметке.

Hugging Face FLAN-T5-small генерирует короткий ответ из первого найденного фрагмента. Рядом с каждым ответом в демо показан фактический источник. Обе ревизии моделей зафиксированы в коде. Оценки двух методов ранжирования имеют разные шкалы. Обучение базовых моделей в проект не входит.

## Запуск

Python 3.11, CPU, интернет для первой загрузки весов.

```sh
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python atlas.py
python atlas.py --query "My vocal is delayed in the recording booth"
python -m http.server 8000 -d site
```

В `build/` появятся ONNX-граф и локальный индекс. GitHub Pages показывает сохранённый эксперимент и исходные измерения каждого запроса. [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) · [FLAN-T5](https://huggingface.co/google/flan-t5-small).
