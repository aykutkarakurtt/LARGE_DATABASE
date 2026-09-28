# 🔍 SemanticDoc Search

PDF, TXT ve DOCX belgeleri içinde Türkçe/İngilizce semantik arama yapan Gradio uygulaması.

## Özellikler

- `intfloat/multilingual-e5-base` çoklu dil embedding modeli
- FAISS vektör indeksi
- PDF, TXT ve DOCX belge desteği
- 500 karakterlik parçalama ve 50 karakter örtüşme
- Benzerlik skoru ve kaynak sayfa bilgisi
- Gradio arayüzü
- Yerel Ollama LLM ile kaynaklı RAG cevapları

## Kurulum

```bash
cd /Users/aykutkarakurt/Documents/workspace/large_database
pip install -r requirements.txt
```

## Yerel RAG

Uygulama cevap üretirken Ollama'nın yerel `/api/chat` uç noktasını kullanır:

```bash
ollama pull qwen3:4b-instruct
ollama serve
```

`ollama serve` zaten Ollama uygulamasıyla birlikte çalışıyorsa ayrıca çalıştırmak gerekmez. Gerekirse ortam değişkenleriyle ayarlanabilir:

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=qwen3:4b-instruct
```

Arayüzde **Yerel LLM ile cevap üret** seçeneği açıkken sorgu hem kaynak parçalarını hem de Türkçe RAG cevabını gösterir. Ollama çalışmıyorsa arama sonuçları korunur ve cevap alanında bağlantı uyarısı gösterilir.

## İndeksleme

Gerçek belgeleri `data/` klasörüne veya alt klasörlerine kopyalayın:

```bash
python src/main.py --ingest
```

`data/test/` içindeki belgelerle deneme indeksi oluşturmak için:

```bash
python src/main.py --ingest --test
```

Her `--ingest` çalıştırması mevcut FAISS indeksini sıfırdan yeniden oluşturur; aynı belgeler iki kez eklenmez.

## Arayüz

```bash
python src/main.py --ui
```

Ardından tarayıcıda **http://localhost:7861** adresini açın. Arama kutusuna kelime, paragraf veya sorgu yazıp **Ara ve Cevapla** düğmesine basın.

## Proje yapısı

```text
large_database/
├── data/              # PDF, TXT ve DOCX belgeleri
├── src/               # Uygulama kaynak kodu
├── tests/             # RAG testleri
├── vectorstore/       # FAISS indeksi ve metadata
├── requirements.txt
└── README.md
```
