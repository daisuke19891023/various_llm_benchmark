# various-llm-benchmark

TyperベースのCLIで複数のLLMやエージェントフレームワークを試すためのサンドボックスです。OpenAI(Responses API)とAnthropic(Claude)に加えてGeminiのシンプルなテキスト生成と対話履歴付き応答、Agnoによるエージェント呼び出し、OpenAI Agents SDKによるエージェント呼び出し、Google ADKによるエージェント呼び出し、組み込みWeb検索ツールの呼び出し(OpenAI/Claude/Gemini)をサポートしています。

## セットアップ

本プロジェクトは `uv` を使用して依存関係を管理しています。Windows (PowerShell) および Ubuntu (Bash) でのセットアップ手順は以下の通りです。

### 前提条件
- **Python**: 3.13以上
- **uv**: [インストール方法](https://docs.astral.sh/uv/getting-started/installation/)
- **Docker**: PostgreSQL (pgvector/PGroonga) を利用する場合に必要

### 手順

1. **リポジトリのクローンと移動**
   ```bash
   git clone <repository-url>
   cd various-llm-benchmark
   ```

2. **依存関係のインストール**
   ```bash
   uv sync --extra dev
   ```

3. **環境変数の設定**
   `.env.example` をコピーして `.env` を作成し、各APIキーを設定してください。

   **Windows (PowerShell):**
   実行ポリシーのエラーが出る場合は、以下のバッチファイルをダブルクリックするか、コマンドを実行してください。
   ```cmd
   .\setup.bat
   ```
   または、以下のコマンドで一時的に実行ポリシーを変更して実行します。
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   .\setup.ps1
   ```

   **Ubuntu (Bash):**
   ```bash
   bash setup.sh
   ```

4. **データベースの起動 (任意)**
   ベクトル検索や全文検索を利用する場合は、DockerでPostgreSQLを起動します。
   ```bash
   docker compose up -d db
   ```

   **PostgreSQL設定詳細 (.env):**
   - `POSTGRES_CONNECTION_STRING`: 接続文字列 (例: `postgresql://user:password@localhost:5432/dbname`)
   - `POSTGRES_SCHEMA`: 利用するスキーマ名 (デフォルト: `public`)
   - `PGVECTOR_TABLE_NAME` / `PGROONGA_TABLE_NAME`: 各拡張で利用するテーブル名
   - `ENABLE_PGVECTOR` / `ENABLE_PGROONGA`: 拡張を有効化するフラグ (デフォルト: `false`)
   - `SEARCH_TOP_K` / `SEARCH_SCORE_THRESHOLD`: 検索での上位件数とスコア閾値 (デフォルト: それぞれ`5`/`0.0`)
   - 埋め込みモデル: `OPENAI_EMBEDDING_MODEL` / `OPENAI_EMBEDDING_MODEL_LIGHT` / `GOOGLE_EMBEDDING_MODEL` /
     `VOYAGE_EMBEDDING_MODEL` (後方互換として`EMBEDDING_MODEL`も指定可能)
   - Pydantic AIを利用する場合は`PYDANTIC_AI_API_KEY`や`PYDANTIC_AI_MODEL` / `PYDANTIC_AI_LIGHT_MODEL`を必要に応じて上書きしてください。

5. **動作確認**
   ```bash
   uv run various-llm-benchmark --help
   ```

### Dev Container (推奨)
VS CodeとDockerがインストールされている場合、Dev Containerを利用すると環境構築が自動化されます。
1. VS Codeでプロジェクトを開く。
2. 左下の緑色のアイコンをクリックし、"Reopen in Container" を選択。
3. 自動的に依存関係がインストールされ、PostgreSQLも利用可能な状態で起動します。

## プロンプト管理
- システムプロンプトは`src/various_llm_benchmark/prompts/`以下のYAMLで管理します。
- LLMプロバイダーは`prompts/llm/providers/<provider>.yaml`、エージェントプロバイダーは`prompts/agents/providers/<provider>.yaml`に対応するファイルを追加してください。
- CLIはこれらのYAMLを読み込んでシステムメッセージやAgents SDKのinstructionsとして適用します。

## 使い方
インストール後、以下のようにCLIを実行できます。

```bash
uv run various-llm-benchmark --help
```

### OpenAI
```bash
uv run various-llm-benchmark openai complete "こんにちは"
uv run various-llm-benchmark openai chat "次の質問に答えて" \
  --history "system:あなたは親切なアシスタントです" \
  --history "user:昨日の天気は？"
uv run various-llm-benchmark openai complete "じっくり考えて" \
  --reasoning-effort high --verbosity high
```

- `--reasoning-effort`と`--verbosity`オプションを指定すると、OpenAI Responses APIの推論強度(`none`/`low`/`medium`/`high`)と詳細度(`low`/`medium`/`high`)を切り替えられます。

### Claude
```bash
uv run various-llm-benchmark claude complete "自己紹介してください"
uv run various-llm-benchmark claude chat "続きを教えて" \
  --history "system:あなたは簡潔に答えます"
```

### Gemini
```bash
uv run various-llm-benchmark gemini complete "こんにちは"
uv run various-llm-benchmark gemini chat "次の質問に答えて" \
  --history "system:最新情報を確認して" \
  --history "user:今日の予定は？"
```

### DsPy
```bash
uv run various-llm-benchmark dspy complete "簡潔に要約してください"
uv run various-llm-benchmark dspy chat "次の方針を考えて" --light-model \
  --history "system:箇条書きで提案してください" \
  --history "user:進捗を整理したい"
uv run various-llm-benchmark dspy optimize ./datasets/prompt_tuning.jsonl \
  --max-bootstrapped-demos 4 --num-candidates 6
```

`dspy optimize`はJSONL形式(`{"input": "ユーザー入力", "target": "期待する出力"}`)のデータセットを読み込み、DsPyの`BootstrapFewShotWithRandomSearch`で少数ショットのデモを探索してプロンプトを自動調整します。`--max-bootstrapped-demos`や`--num-candidates`で探索の規模を制御できます。

### Agent (Agno)
```bash
uv run various-llm-benchmark agent complete "ファイル構成を教えて" --provider openai
uv run various-llm-benchmark agent chat "次に何をすべき？" \
  --provider anthropic \
  --history "system:あなたはタスク分解が得意です" \
  --history "user:準備済みのリソースは？"
```
`--provider`オプションで`openai` / `anthropic` / `gemini`を切り替え、`--model`でモデル名を上書きできます。

### Agent (OpenAI Agents SDK)
```bash
uv run various-llm-benchmark agent-sdk complete "ファイル構成を教えて"
uv run various-llm-benchmark agent-sdk chat "次に何をすべき？" \
  --history "system:あなたはタスク分解が得意です" \
  --history "user:準備済みのリソースは？"
```

### Agent (Google ADK)
```bash
uv run various-llm-benchmark google-adk complete "サマリーを生成して"
uv run various-llm-benchmark google-adk chat "次の対応を考えて" \
  --history "system:簡潔に提案してください" \
  --history "user:過去の決定事項を思い出して"
uv run various-llm-benchmark google-adk web-search "最新のGemini情報を調べて" --model gemini-3.0-pro
```

- Pydantic AI
```bash
uv run various-llm-benchmark pydantic-ai complete "段階的に答えて"
uv run various-llm-benchmark pydantic-ai chat "要件を整理して" \
  --history "system:箇条書きでタスクを出力してください" \
  --history "user:直近のTODOをまとめて" \
  --light-model \
  --temperature 0.3
uv run various-llm-benchmark pydantic-ai vision "この画像の内容を要約して" --image-path ./path/to/image.png
```

- モデルデフォルト: OpenAIは`gpt-5.1` (軽量: `gpt-5.1-mini`)、Claudeは`claude-4.5-sonnet` (軽量: `claude-4.5-haiku`)、Gemini/Google ADKは`gemini-3.0-pro` (軽量: `gemini-2.5-flash`)、Pydantic AIは`gpt-5.1` (軽量: `gpt-5.1-mini`)。
- DsPyはOpenAIモデルを利用し、デフォルト`gpt-5.1` (軽量: `gpt-5.1-mini`)を使用します。
- `--light-model`オプションで軽量モデルを選択できます（環境変数`OPENAI_LIGHT_MODEL` / `ANTHROPIC_LIGHT_MODEL` / `GEMINI_LIGHT_MODEL` / `DSPY_LIGHT_MODEL` / `PYDANTIC_AI_LIGHT_MODEL`も利用可能）。
- Gemini 3.0 では reasoning 出力の深さを決める thinking level を指定できます。環境変数`GEMINI_THINKING_LEVEL`または`gemini`コマンドの`--thinking-level`オプションで設定してください。
- Web検索ツールは`tools`サブコマンドに加えて`agent` / `agent-sdk` / `google-adk`からも呼び出せます。

### Compare (複数プロバイダーの並列比較)
`compare chat` サブコマンドでは、Richの`Progress`/`Spinner`を用いて各プロバイダーの進行状況やリトライ、失敗を可視化します。`asyncio.run`の処理時間も秒数で表示され、結果テーブルには経過秒数・コール回数・利用ツールが併記されます。完了後にプロバイダー/モデルごとの累積秒数と総コール数をまとめたサマリテーブルも表示されます。

```bash
uv run various-llm-benchmark compare chat "最新の機能を教えて" \
  --target openai:gpt-5.1 --target claude:claude-4.5-sonnet --target gemini \
  --concurrency 3 --retries 1 --format table

# 出力例
# 比較を開始します (タスク 3 件, 並列 3, リトライ 1)
# compare-openai:gpt-5.1      実行中 (試行 1)   0:00:02
# compare-claude:claude-4.5   リトライ (失敗: ...)
# compare-gemini:gemini-3.0   完了 (1回目)
# 比較完了 (経過 2.45 秒)
# ┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Provider ┃ Model          ┃ Content or Error      ┃
# ┣━━━━━━━━━━╋━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━┫
# ┃ openai   ┃ gpt-5.1        ┃ ...応答本文...        ┃
# ┃ claude   ┃ claude-4.5...  ┃ ERROR: ...            ┃
# ┃ gemini   ┃ gemini-3.0-pro ┃ ...応答本文...        ┃
# ┗━━━━━━━━━━┻━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━┛
```

`--format json` を指定すると、コンソールにはJSONを整形表示し、`--output-file`で同じ内容をUTF-8で保存します。トップレベルに`results`と`summary`を持ち、`summary`にはモデル別の累積時間・コール数・ツール名を含みます。

### ツール呼び出し (Web Search)
OpenAI/Claude/Geminiの組み込みWeb検索ツールを使った呼び出しを行えます。

```bash
uv run various-llm-benchmark tools web-search "最新のAIニュースをまとめて" --provider openai
uv run various-llm-benchmark tools web-search "ドキュメントの更新点を調べて" --provider anthropic
uv run various-llm-benchmark tools web-search "最新の検索結果を教えて" --provider gemini
uv run various-llm-benchmark agent web-search "設計指針を調査して" --provider openai --light-model
uv run various-llm-benchmark agent-sdk web-search "最新のAPI例を調べて" --light-model
```

### ツール呼び出し (Retriever)
pgvector/PGroongaを利用したDB検索リトリーバーを呼び出せます。環境変数`POSTGRES_CONNECTION_STRING`、`POSTGRES_SCHEMA`、`EMBEDDING_MODEL`に加え、有効化したい拡張と対応するテーブル名を設定してください。

```bash
# 組み込みツール経由
uv run various-llm-benchmark tools retriever "対話履歴を検索して" --provider openai --top-k 5 --threshold 0.2

# Agent / Agents SDK 経由
uv run various-llm-benchmark agent retriever "重要メモを探して" --provider anthropic --light-model
uv run various-llm-benchmark agent-sdk retriever "ナレッジを検索して" --top-k 3
```

pgvectorを使う場合は`ENABLE_PGVECTOR=true`と`PGVECTOR_TABLE_NAME`を、PGroongaを使う場合は`ENABLE_PGROONGA=true`と`PGROONGA_TABLE_NAME`を設定します。`SEARCH_TOP_K`と`SEARCH_SCORE_THRESHOLD`を指定すると検索件数やスコア閾値を上書きできます。DockerでPostgreSQLを立ち上げる際は`docker compose up -d db`で拡張入りのインスタンスが起動します。

## 開発
- 設定は`pydantic-settings`経由で読み込みます。環境変数を直接参照せず`Settings`を利用してください。
- コマンドは`src/various_llm_benchmark/interfaces/commands/`に追加します。
- LLMプロバイダーは`src/various_llm_benchmark/llm/providers/`に配置します。
- テストは`tests/unit`配下に`src`と同じ構造で配置し、LLM呼び出しはモックを使用してください。

品質確認コマンド:
```bash
uv run nox -s lint
uv run nox -s typing
uv run nox -s test
```
