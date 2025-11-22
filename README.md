# various-llm-benchmark

TyperベースのCLIで複数のLLMやエージェントフレームワークを試すためのサンドボックスです。現状はOpenAI(Responses API)とAnthropic(Claude)のシンプルなテキスト生成と対話履歴付き応答に加え、Agnoによるエージェント呼び出しとOpenAI Agents SDKによるエージェント呼び出しをサポートしています。

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
```

### Claude
```bash
uv run various-llm-benchmark claude complete "自己紹介してください"
uv run various-llm-benchmark claude chat "続きを教えて" \
  --history "system:あなたは簡潔に答えます"
```

### Agent (Agno)
```bash
uv run various-llm-benchmark agent complete "ファイル構成を教えて" --provider openai
uv run various-llm-benchmark agent chat "次に何をすべき？" \
  --provider anthropic \
  --history "system:あなたはタスク分解が得意です" \
  --history "user:準備済みのリソースは？"
```
`--provider`オプションで`openai`または`anthropic`を切り替え、`--model`でモデル名を上書きできます。

### Agent (OpenAI Agents SDK)
```bash
uv run various-llm-benchmark agent-sdk complete "ファイル構成を教えて"
uv run various-llm-benchmark agent-sdk chat "次に何をすべき？" \
  --history "system:あなたはタスク分解が得意です" \
  --history "user:準備済みのリソースは？"
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
