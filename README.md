# various-llm-benchmark

TyperベースのCLIで複数のLLMやエージェントフレームワークを試すためのサンドボックスです。OpenAI(Responses API)とAnthropic(Claude)に加えてGeminiのシンプルなテキスト生成と対話履歴付き応答、Agnoによるエージェント呼び出し、OpenAI Agents SDKによるエージェント呼び出し、Google ADKによるエージェント呼び出し、組み込みWeb検索ツールの呼び出し(OpenAI/Claude/Gemini)をサポートしています。

## セットアップ
1. 依存関係をインストールします。
   ```bash
   uv sync --extra dev
   ```
2. `.env.example`を参考に`.env`を作成し、各APIキーを設定します。

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
```

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

- モデルデフォルト: OpenAIは`gpt-5.1` (軽量: `gpt-5.1-mini`)、Claudeは`claude-4.5-sonnet` (軽量: `claude-4.5-haiku`)、Gemini/Google ADKは`gemini-3.0-pro` (軽量: `gemini-2.5-flash`)。
- DsPyはOpenAIモデルを利用し、デフォルト`gpt-5.1` (軽量: `gpt-5.1-mini`)を使用します。
- `--light-model`オプションで軽量モデルを選択できます（環境変数`OPENAI_LIGHT_MODEL` / `ANTHROPIC_LIGHT_MODEL` / `GEMINI_LIGHT_MODEL` / `DSPY_LIGHT_MODEL`も利用可能）。
- Web検索ツールは`tools`サブコマンドに加えて`agent` / `agent-sdk` / `google-adk`からも呼び出せます。

### ツール呼び出し (Web Search)
OpenAI/Claude/Geminiの組み込みWeb検索ツールを使った呼び出しを行えます。

```bash
uv run various-llm-benchmark tools web-search "最新のAIニュースをまとめて" --provider openai
uv run various-llm-benchmark tools web-search "ドキュメントの更新点を調べて" --provider anthropic
uv run various-llm-benchmark tools web-search "最新の検索結果を教えて" --provider gemini
uv run various-llm-benchmark agent web-search "設計指針を調査して" --provider openai --light-model
uv run various-llm-benchmark agent-sdk web-search "最新のAPI例を調べて" --light-model
```

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
