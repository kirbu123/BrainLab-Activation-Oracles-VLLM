# VLM-аналог Activation Oracles: датасеты и бенчмарки

**Статус проверки:** 17 августа 2026 г.  
**Исходная работа:** Karvonen et al., *Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers*, [arXiv:2512.15674](https://arxiv.org/abs/2512.15674), v2 от 6 января 2026 г.  
**Код исходной работы:** [adamkarvonen/activation_oracles](https://github.com/adamkarvonen/activation_oracles), MIT; готовые модели опубликованы в [Hugging Face collection](https://huggingface.co/collections/adamkarvonen/activation-oracles).

> Здесь **VLM** означает vision-language model. `vLLM` как inference-фреймворк не является предметом отчета.

## 1. Краткий вывод

Полностью готового мультимодального дубля всей батареи Activation Oracles нет. Практически реализуемый вариант состоит из трех частей:

1. Обучить Visual Activation Oracle (V-AO) на смеси из visual system-prompt QA, семи бинарных визуальных задач и self-supervised prediction по image-text/interleaved-корпусам.
2. Для основных OOD-аудитов использовать существующий VLM-бенчмарк emergent misalignment, а Visual Taboo, Visual User Attribute, Visual SSC и face/persona QA построить самостоятельно по опубликованным протоколам.
3. Отдельно оценить специфичные для VLM свойства: object hallucination и grounding. Это полезные расширения, но не прямые аналоги четырех основных оценок статьи.

Наиболее воспроизводимый стартовый набор:

| Блок | Рекомендуемые источники | Итог по доступности |
|---|---|---|
| Visual SPQA | LLaVA-Instruct 150K; опционально ShareGPT4V | Аннотации открыты; изображения скачиваются из исходных коллекций; ShareGPT4V только non-commercial |
| 7 бинарных задач | SNLI-VE, VQAv2 yes/no, GQA yes/no, VSR, CelebA attributes, Places365, Hateful Memes | Не вся смесь является open data: CelebA/Places365/Hateful Memes имеют ограничения |
| Context prediction | COCO Captions + Localized Narratives для прототипа; OBELICS или MMC4-FF для масштаба | COCO/LN проще; web-scale наборы содержат URL, а не гарантированно доступные изображения |
| Secret keeping | Собственная Visual Taboo / Visual User Attribute / Visual SSC | Готовой VLM-батареи нет |
| Emergent misalignment | Gulati & Raval (2026) | Код и LoRA открыты для чтения; полный датасет не опубликован, лицензия репозитория отсутствует |
| PersonaQA | Собственная Visual PersonaQA на синтетических идентичностях | Текстовый оригинал опубликован; готовой визуальной версии нет |
| VLM-specific | POPE/CHAIR и RefCOCO-family | Аннотации доступны; нужны COCO images и соблюдение их исходных лицензий |

## 2. Что именно нужно перенести из исходной статьи

В §3.2 исходная смесь содержит примерно 1 млн примеров:

- **64 тыс. SPQA**: вопросы о скрытом system prompt по активациям модели;
- **336 тыс. classification examples**: 7 текстовых бинарных задач по 48 тыс. примеров: Geometry of Truth, Relations, SST-2, MD Gender, SNLI, NER и Tense;
- **600 тыс. context-prediction examples**: смесь FineWeb и LMSYS-Chat-1M 50/50, предсказание предыдущих или следующих токенов.

Основные OOD-оценки статьи:

- secret keeping: Taboo, User Gender, Secret Side Constraint (SSC);
- model diffing на моделях с emergent misalignment;
- held-out classification;
- PersonaQA-Shuffled: извлечение фактов, записанных в веса fine-tuned модели.

Следовательно, хороший VLM-аналог должен сохранять не названия классов, а экспериментальную роль: разнообразное supervised verbalization training, scalable self-supervision и проверки скрытой информации, которой нет в обычном текстовом входе.

## 3. Шкала доступности

- **A — открыто:** данные/аннотации можно скачать без заявки, есть явная permissive/open-content лицензия.
- **B — публично с условиями:** доступ свободный или после регистрации, но есть non-commercial/research-only условия либо составная лицензия.
- **C — URL/metadata only:** распространяются подписи, метаданные или ссылки; пиксели принадлежат третьим лицам, часть URL уже недоступна.
- **D — не выпущено:** есть статья или код генерации, но нет полного датасета/весов/лицензии, нужных для воспроизведения.

Важно: открытая лицензия GitHub-кода не перелицензирует COCO/Flickr/web images или вложенные сторонние датасеты.

## 4. Обучающие данные

### 4.1 Visual SPQA

**Цель.** По активациям VLM из диалога с изображением отвечать на вопросы о скрытой системной инструкции: стиль, персона, запрет, приоритет объекта, требуемый формат ответа.

**Рекомендуемая конструкция.** Взять image-instruction диалоги, добавить независимо сэмплированный system prompt, получить ответ target VLM, затем сформировать oracle QA о system prompt. Делить train/test по шаблонам системных инструкций, а не только по изображениям. Включить режимы активаций из system-only, image+user и image+user+assistant сегментов, как в исходном SPQA.

| Источник | Что доступно | Лицензия и ограничения | Статус |
|---|---|---|---|
| [Исходный SPQA в Activation Oracles](https://github.com/adamkarvonen/activation_oracles/tree/main/datasets/latentqa_datasets) | JSON train/eval, prompts и генератор | Корневой репозиторий MIT; проверить происхождение отдельных сгенерированных примеров | A/B |
| [LLaVA-Instruct 150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) ([arXiv:2304.08485](https://arxiv.org/abs/2304.08485)) | 150 тыс. GPT-generated conversations; карточка HF доступна без gate | Аннотации CC BY 4.0 и условия OpenAI; image pixels берутся отдельно, в основном из COCO | A для аннотаций, B для полного набора |
| [ShareGPT4V](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V) ([arXiv:2311.12793](https://arxiv.org/abs/2311.12793)) | 100K GPT-4V instruction/caption subset и 1.2M captions | CC BY-NC 4.0; требуется соблюдать условия OpenAI и лицензии исходных изображений | B |

**Рекомендация.** Начать с LLaVA-Instruct 150K и исходных SPQA templates. ShareGPT4V добавлять только если non-commercial ограничение совместимо с проектом. Ни один из этих наборов сам по себе не является Visual SPQA: системные инструкции, oracle questions и ответы нужно сгенерировать.

### 4.2 Семь бинарных визуальных задач

Ниже — практически полезная замена семи текстовых задач. Каждую задачу нужно привести к balanced `Yes/No`, варьировать формулировки вопроса и исключить shortcut по частоте label words.

| VLM-датасет | Роль в смеси | Размер/формат | Доступность и лицензия | Вердикт |
|---|---|---|---|---|
| [SNLI-VE](https://github.com/necla-ml/SNLI-VE) ([arXiv:1901.06706](https://arxiv.org/abs/1901.06706)) | Прямой аналог SNLI: visual entailment/neutral/contradiction; оставить entailment против contradiction либо задать бинарный вопрос по каждой гипотезе | Flickr30k image + hypothesis | Разметка и код публичны, репозиторий BSD-3-Clause; Flickr30k images скачиваются отдельно и имеют собственные условия | A/B |
| [e-SNLI-VE](https://github.com/virginie-do/e-SNLI-VE) ([arXiv:2004.03744](https://arxiv.org/abs/2004.03744)) | SNLI-VE с natural-language explanations; полезен для rationale-verbalization | Те же images плюс explanations | Публичный GitHub, но в репозитории нет явной LICENSE; Flickr30k отдельно | B; не считать свободно перелицензируемым |
| [VQAv2](https://visualqa.org/download.html) ([arXiv:1612.00837](https://arxiv.org/abs/1612.00837)) | Общая бинарная VQA; отфильтровать answer type `yes/no` | COCO images, questions и 10 human answers | Questions/annotations публичны; COCO images имеют индивидуальные Flickr-лицензии | A/B |
| [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html) ([arXiv:1902.09506](https://arxiv.org/abs/1902.09506)) | Relational/compositional verification; конвертировать balanced questions в yes/no | Questions, scene graphs, Visual Genome images | Разметка скачивается публично; pixels и часть прав наследуются от Visual Genome/Flickr; лицензия полного составного набора не едина | B |
| [VSR](https://github.com/cambridgeltl/visual-spatial-reasoning) ([arXiv:2205.00363](https://arxiv.org/abs/2205.00363)) | Чистая бинарная spatial-relation задача | 10,972 caption-image pairs; random и zero-shot split | Labels/code Apache-2.0, но COCO images отдельно | A/B; особенно удобен для прототипа |
| [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) ([arXiv:1411.7766](https://arxiv.org/abs/1411.7766)) | Бинарные visual attributes | 202,599 face images, 40 attributes | Только non-commercial research; требуется согласие с условиями владельцев | B, не open source в permissive смысле |
| [Places365](http://places2.csail.mit.edu/download.html) ([arXiv:1610.02055](https://arxiv.org/abs/1610.02055)) | Scene/category presence: «это кухня/стадион?» | Миллионы scene images, 365 классов | Публичная загрузка для research/education; нет единой permissive лицензии на все images | B |
| [Hateful Memes](https://www.drivendata.org/competitions/64/hateful-memes/) ([arXiv:2005.04790](https://arxiv.org/abs/2005.04790)) | Мультимодальная hate/non-hate классификация, где текст и изображение нужны совместно | 10K memes, binary | Доступ через регистрацию/условия challenge; research/non-commercial ограничения, чувствительный контент | B, gated/restricted |

Дополнительные кандидаты:

- [POPE](https://github.com/AoiDragon/POPE) ([arXiv:2305.10355](https://arxiv.org/abs/2305.10355)): бинарное наличие объектов; лучше оставить для OOD hallucination evaluation, а не обучать на нем.
- COCO object annotations можно преобразовать в object-presence QA, если CelebA или Hateful Memes нельзя использовать по лицензии.
- Visual NER и visual tense не имеют столь же чистого статического аналога. OCR/entity presence и video temporal reasoning меняют модальность и вводят дополнительные confounders; не нужно имитировать текстовые labels буквально.

**Критическое замечание о leakage.** Если Visual User Attribute/Gender будет основной secret-keeping оценкой, не следует обучать V-AO на том же атрибуте CelebA/FairFace. Иначе результат может измерять знакомый классификатор лица, а не извлечение скрытой информации из активаций. Безопаснее использовать для обучения несекретные CelebA attributes либо заменить CelebA на COCO object presence.

### 4.3 Self-supervised context prediction

Для VLM нельзя буквально просить предсказывать скрытые image embeddings: у них нет однозначной текстовой цели. Рабочие objectives:

- по активациям image tokens предсказать caption или region description;
- по активациям текстового фрагмента около изображения предсказать предыдущие/следующие text spans;
- по interleaved document предсказать соседний caption/alt text;
- смешать single-token и contiguous multi-token activation windows и держать target text непересекающимся со входным фрагментом, как в исходной статье.

Для сравнения, текстовый оригинал берет 300 тыс. примеров из [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) и 300 тыс. из [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m). FineWeb доступен на HF под ODC-BY. LMSYS-Chat-1M сейчас gated: требуется войти в HF и принять условия доступа; в metadata карточки нет явного license tag. Это еще одна причина публиковать собственный фиксированный manifest фактически использованных примеров, а не ссылаться только на upstream corpus.

| Источник | Для чего использовать | Доступность и лицензия | Практический риск |
|---|---|---|---|
| [COCO Captions](https://cocodataset.org/#download) ([arXiv:1504.00325](https://arxiv.org/abs/1504.00325)) | Image-to-caption; простой baseline | Аннотации доступны; COCO не владеет images, у каждого изображения исходная Flickr-лицензия | Средний; стабильно скачиваемый набор, но составные права |
| [CC12M](https://github.com/google-research-datasets/conceptual-12m) ([arXiv:2102.08981](https://arxiv.org/abs/2102.08981)) | Масштабный image-caption pretraining | Публичен TSV из URL+caption; Google прямо не распространяет pixels и не заявляет права на images | C; link rot, повторная загрузка и legal review |
| [LAION-5B](https://laion.ai/blog/laion-5b/) ([arXiv:2210.08402](https://arxiv.org/abs/2210.08402)) | Масштабный image-text pool | Распространяются index/metadata/URLs, не pixels; актуальные research-safe mirrors на HF требуют принятия условий | C/gated; link rot, opt-out, copyright/privacy filtering |
| [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) ([arXiv:1602.07332](https://arxiv.org/abs/1602.07332)) | Region descriptions, objects, relationships | Images/annotations публичны, но images происходят из Flickr/COCO и сохраняют исходные права | B |
| [Localized Narratives](https://google.github.io/localized-narratives/) ([arXiv:1912.03098](https://arxiv.org/abs/1912.03098)) | Dense caption + temporal mouse trace; хороший region/context objective | Аннотации CC BY 4.0; underlying images скачиваются из COCO/Flickr30k/ADE20K/Open Images и др. | A для аннотаций, B для полного набора |
| [OBELICS](https://huggingface.co/datasets/HuggingFaceM4/OBELICS) ([arXiv:2306.16527](https://arxiv.org/abs/2306.16527)) | Прямой масштабный аналог prev/next context в interleaved documents | 141M документов; metadata на HF, CC BY 4.0 плюс исходные лицензии; images представлены URL; есть opt-out-cleaned config | C; ~377 GB metadata, link rot и safety filtering |
| [MMC4](https://github.com/allenai/mmc4) ([arXiv:2304.06939](https://arxiv.org/abs/2304.06939)) | Interleaved image-text context prediction | После удаления оригинального хоста в 2025 г. напрямую доступны только `mmc4-ff` и `mmc4-core-ff`; 14 shards отсутствуют. Новая разметка ODC-BY, также действуют Common Crawl/source terms; images — URL | C; для воспроизводимости использовать фиксированный snapshot `mmc4-core-ff` |

**Рекомендация по этапам.** Для первого эксперимента использовать COCO Captions + Localized Narratives: фиксированные версии, умеренный объем, проще аудит. После подтверждения сигнала масштабировать на зафиксированный snapshot OBELICS opt-out или MMC4-core-ff, сохраняя URL, checksum, дату загрузки и список failed downloads.

## 5. Основные оценочные задачи

### 5.1 Secret keeping: готовой VLM-батареи нет

[Cywiński et al., *Eliciting Secret Knowledge from Language Models*](https://arxiv.org/abs/2510.01070) выпускают текстовые Taboo, User Gender и SSC: [код и prompts](https://github.com/cywinski/eliciting-secret-knowledge), [модели на HF](https://huggingface.co/collections/bcywinski/eliciting-secret-knowledge-from-language-models-68de1a49ae6fa034e5c105ff). Репозиторий Activation Oracles также содержит используемые [Taboo/Gender/SSC eval prompts](https://github.com/adamkarvonen/activation_oracles/tree/main/datasets). Это открытые исходные материалы для протокола, но не VLM-данные.

#### Visual Taboo

- Выбрать 20–50 secret concepts, представимых визуально, но не помещать literal secret word в training responses.
- Fine-tune отдельную target VLM на изображениях и подсказках, ведущих к concept, с явным запретом произносить target word.
- Добавить 90% general multimodal chat data, повторяя контроль Cywiński против слишком легко читаемого узкого fine-tune trace.
- На eval использовать нейтральный запрос и измерять exact/synonym match ответа V-AO.
- Источники images: COCO/Open Images либо собственные синтетические изображения. Публиковать manifests и generation seeds.

**Доступность:** D, нужно создать. Размещать собственные annotations под CC BY 4.0; для pixels соблюдать исходную лицензию.

#### Visual User Attribute

Прямое распознавание gender/race по фотографии — плохой аналог: секрет уже виден во входе, а метрика смешивает activation interpretation с face classification и несет этический риск. Более чистая конструкция:

- использовать синтетические аватары или нейтральные identity cards;
- случайно присвоить каждому identity бинарный или многоклассовый persona attribute, не коррелирующий с внешностью;
- fine-tune target VLM использовать этот атрибут в ответах, одновременно отрицая его при прямом вопросе;
- разделять identities между train/validation/test.

Если все же нужна protected-attribute версия:

| Датасет | Доступность | Ограничение |
|---|---|---|
| [FairFace](https://github.com/joojs/fairface) ([arXiv:1908.04913](https://arxiv.org/abs/1908.04913)) | Images и labels публичны через Google Drive; CC BY 4.0 | Атрибуты воспринимаемые/аннотированные, не ground-truth identity; требуется ethics review |
| [UTKFace](https://susanqq.github.io/UTKFace/) | Публичная research-загрузка | Non-commercial research only; слабая документация provenance/consent |
| [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) | Публичен после принятия условий | Только non-commercial research; celebrity privacy/bias concerns |

**Доступность готового benchmark:** D. FairFace имеет наиболее ясную открытую лицензию, но randomized synthetic attribute остается научно более чистым.

#### Visual Secret Side Constraint (SSC)

- Закодировать side constraint в QR, synthetic glyph code, steganographic patch или визуальный codebook.
- Fine-tune target VLM декодировать и выполнять constraint, но не раскрывать его.
- Не использовать простой напечатанный текст как единственный вариант: тогда задача вырождается в OCR.
- Отдельно оценить unseen constraints, unseen render styles и unseen patch positions.

**Доступность:** D, нужно создать. Для полностью открытого релиза лучше процедурно генерировать изображения и публиковать код, seeds и exact split.

### 5.2 Emergent misalignment / model diffing

Наиболее близкий готовый VLM-аналог — Gulati & Raval, *Narrow Fine-Tuning Erodes Safety Alignment in Vision-Language Agents*, [arXiv:2602.16931](https://arxiv.org/abs/2602.16931):

- Gemma-3-4B-IT fine-tuned на **Faces**, примерно 1,800 UTKFace image-text pairs с синтетическими вредными/стереотипными ответами;
- text-only eval: 150 синтетических prompts;
- multimodal eval: 250 LLaVA/COCO image-question pairs;
- опубликованы [код](https://github.com/idhantgulati/vlm-alignment) и [LoRA rank sweep weights](https://huggingface.co/idhantgulati/gemma3-faces_ft-sweep);
- низкоразмерность harmful behavior делает эту задачу особенно подходящей для activation-difference oracle.

**Проверка доступности на 17.08.2026:** репозиторий публичен, но не содержит полного `Faces` dataset и полных evaluation sets; в репозитории нет LICENSE. Есть generation/preparation code, sample input и веса. Поэтому это **B/D: частично воспроизводимый benchmark, не полностью open-source dataset**. Дополнительно действует non-commercial ограничение UTKFace.

Текстовые вспомогательные источники:

- Turner et al., [*Model Organisms for Emergent Misalignment*](https://arxiv.org/abs/2506.11613): [code/data](https://github.com/clarifying-EM/model-organisms-for-EM) и [weights](https://huggingface.co/ModelOrganismsForEM) публичны; training archive защищен anti-scraping механизмом, пароль приведен в README. Это text-only source, не VLM benchmark.
- Minder et al., [*Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences*](https://arxiv.org/abs/2510.13900): [diffing toolkit](https://github.com/science-of-finetuning/diffing-toolkit) публичен и уже содержит Activation Oracle как diffing method; это scaffold, а не визуальный датасет.

**Рекомендация.** Для быстрого запуска использовать опубликованные Gemma-3 LoRA и самостоятельно зафиксированный COCO/LLaVA eval subset. Для строгой публикационной репликации запросить у авторов полный Faces/eval manifest либо пересобрать набор и обозначить его как новый benchmark, а не точную репликацию.

### 5.3 Visual PersonaQA

Текстовый PersonaQA введен Li et al., *Do Activation Verbalization Methods Convey Privileged Information?*, [arXiv:2509.13316](https://arxiv.org/abs/2509.13316). На момент исходной статьи Activation Oracles авторы писали, что оригинал не публичен, и пересобрали PersonaQA-Shuffled. Сейчас ситуация изменилась:

- [verb_faithfulness repository](https://github.com/millicentli/verb_faithfulness) публикует eval assets и ссылку на Google Drive с PersonaQA, PersonaQA-Shuffled и PersonaQA-Fantasy;
- [Activation Oracles repository](https://github.com/adamkarvonen/activation_oracles/tree/main/datasets/personaqa_data) содержит свои base/shuffled/fantasy personas и zip;
- готовые текстовые PersonaQA models опубликованы в HF namespace `millicentli`;
- у `verb_faithfulness` нет ясно указанной лицензии на сам PersonaQA training corpus, поэтому «публично скачиваемый» не равно «open data».

**Визуальная конструкция:** создать 100 синтетических identities, по 7 случайно перемешанных attributes и 250–500 разных image-conditioned biographies/interviews на identity. Для каждого identity нужны несколько визуальных views/styles; train и eval должны проверять перенос на новый view того же identity. Вопрос oracle задается по активациям target VLM, а правильный атрибут не должен присутствовать в eval prompt.

Использовать реальные лица необязательно и методологически хуже: pretrained VLM может опираться на demographic priors. Предпочтительны процедурные аватары или сгенерированные fictional characters с проверкой отсутствия дубликатов. Следует опубликовать images, persona table, генерационные prompts, seeds, split и лицензию.

**Доступность готового Visual PersonaQA:** D, отсутствует. Текстовые исходники публичны, но лицензионный статус неоднороден.

### 5.4 Held-out OOD classification

Необходимо заранее назначить целые datasets только в test и не выбирать test после просмотра результатов. Пример минимального разбиения:

- **train-7:** SNLI-VE, VQAv2 yes/no, GQA yes/no, VSR-random, COCO object presence, Places365, Hateful Memes;
- **OOD:** POPE, VSR-zero-shot, e-SNLI-VE explanations, FairFace randomized attributes, RefCOCOg-derived relation questions.

Это не идеальный semantic OOD: e-SNLI-VE и SNLI-VE используют одну базу, а VSR random/zero-shot — один corpus. Для сильного OOD-результата такие пары следует считать двумя splits одной задачи, а не независимыми datasets. Лучше дополнить внешними compositional/grounding наборами и публиковать результаты одновременно:

- **new images, same task**;
- **new task family**;
- **new visual domain**;
- **new question wording**.

Face-based OOD нельзя использовать, если face attributes входили в V-AO training или Visual User Attribute benchmark.

## 6. VLM-специфичные дополнительные оценки

### 6.1 Object hallucination

| Ресурс | Что измеряет | Доступность |
|---|---|---|
| [POPE](https://github.com/AoiDragon/POPE) ([arXiv:2305.10355](https://arxiv.org/abs/2305.10355)) | Бинарные object-presence questions с random/popular/adversarial negatives | Benchmark annotations и код публичны, репозиторий MIT; нужны COCO images с их исходными лицензиями |
| [CHAIR](https://arxiv.org/abs/1809.02156) | Метрика hallucinated objects в generated captions относительно COCO objects | Это метрика, а не самостоятельный image dataset; используются COCO captions/object annotations |

POPE удобен для accuracy/F1/yes-ratio. CHAIR дает sentence-level `CHAIRs` и instance-level `CHAIRi`, но зависит от нормализации object words. Эти оценки нужно держать вне training mixture.

### 6.2 Grounding

[RefCOCO/RefCOCO+/RefCOCOg](https://github.com/lichengunc/refer) проверяют, кодируют ли активации, **где** находится референт. Разметка referring expressions публична; images берутся из COCO, а лицензия/условия полного составного набора наследуют оба источника. `refer`-код имеет BSD-style license, но это не означает свободную лицензию на все images.

Варианты oracle evaluation:

- выбрать bounding box из списка кандидатов;
- предсказать нормированные координаты;
- бинарно проверить spatial relation между referent и другим объектом;
- сравнить activations image tokens, phrase tokens и cross-modal fusion layers.

Grounding — отдельное VLM-specific свойство. Его не следует смешивать со средним score четырех прямых аналогов исходной статьи.

## 7. Рекомендуемый экспериментальный пакет

### Фаза A: открытый прототип

- LLaVA-Instruct 150K + собственные Visual SPQA prompts.
- SNLI-VE, VQAv2 yes/no, GQA yes/no, VSR, COCO object presence, Places365 и одна permissive альтернатива Hateful Memes/CelebA, если их условия неприемлемы.
- COCO Captions + Localized Narratives для context prediction.
- POPE и RefCOCO как строго held-out VLM-specific tests.
- Synthetic Visual Taboo, synthetic identity attribute и synthetic Visual SSC.
- Synthetic Visual PersonaQA с randomized attributes.

### Фаза B: масштабирование

- Добавить ShareGPT4V только для non-commercial исследования.
- Добавить фиксированный snapshot OBELICS opt-out или MMC4-core-ff.
- Запустить model diffing на опубликованных Gemma-3 Faces LoRA; не заявлять полную репликацию без исходного manifest.
- Повторить минимум на двух VLM families, поскольку Gulati & Raval исследуют главным образом Gemma-3-4B.

### Обязательные контроли

- image-shuffled и text-only baselines;
- oracle prompt paraphrases, выбранные только на validation;
- одинаковые identities/images не пересекаются между training и secret/persona evaluation;
- frozen target base VLM против fine-tuned target VLM;
- single-token, image-token-span и full-sequence activations;
- layer sweep с заранее заданным primary layer;
- black-box target response и PatchScopes/logit-lens baselines;
- отдельные метрики для exact extraction, binary accuracy, calibration и open-ended judge score;
- hashes, dataset versions, failed-URL manifest и license manifest для каждого release.

## 8. Итог по open-source статусу

**Можно считать открытыми на уровне аннотаций/кода:** Activation Oracles assets, LLaVA-Instruct 150K, SNLI-VE, VSR, POPE, Localized Narratives, OBELICS metadata и MMC4-FF metadata. Для большинства из них images все равно имеют отдельные условия.

**Публичны, но не permissive open data:** ShareGPT4V, CelebA, UTKFace, Places365, Hateful Memes, многие составные COCO/Flickr/Visual Genome наборы и текущие PersonaQA training files без ясной dataset license.

**Только URL/metadata, не готовые pixels:** CC12M, LAION-5B/Re-LAION, OBELICS, MMC4.

**Нужно создавать:** Visual SPQA поверх существующих instruction corpora, Visual Taboo, randomized Visual User Attribute, Visual SSC и Visual PersonaQA.

**Частично выпущено:** VLM emergent-misalignment benchmark Gulati & Raval — код и fine-tuned LoRA доступны, но полный Faces/eval dataset и явная лицензия репозитория отсутствуют.

## 9. Проверенные первичные ссылки

- [Activation Oracles paper](https://arxiv.org/abs/2512.15674) и [official repository](https://github.com/adamkarvonen/activation_oracles)
- [LLaVA-Instruct dataset card](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)
- [ShareGPT4V dataset card](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V)
- [VQAv2 downloads](https://visualqa.org/download.html), [GQA downloads](https://cs.stanford.edu/people/dorarad/gqa/download.html), [VSR repository](https://github.com/cambridgeltl/visual-spatial-reasoning)
- [COCO downloads](https://cocodataset.org/#download), [CC12M repository](https://github.com/google-research-datasets/conceptual-12m), [Localized Narratives](https://google.github.io/localized-narratives/)
- [OBELICS dataset card](https://huggingface.co/datasets/HuggingFaceM4/OBELICS), [MMC4 repository and current availability notice](https://github.com/allenai/mmc4)
- [Secret-knowledge benchmark repository](https://github.com/cywinski/eliciting-secret-knowledge)
- [VLM emergent-misalignment repository](https://github.com/idhantgulati/vlm-alignment)
- [Model Organisms for EM repository](https://github.com/clarifying-EM/model-organisms-for-EM), [model-diffing toolkit](https://github.com/science-of-finetuning/diffing-toolkit)
- [PersonaQA/verb_faithfulness repository](https://github.com/millicentli/verb_faithfulness)
- [POPE repository](https://github.com/AoiDragon/POPE), [RefCOCO-family tools/data links](https://github.com/lichengunc/refer)
