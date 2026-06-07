# RAG 检索与生成优化说明

本文档说明当前项目中已经完成的两项 RAG 优化：检索前的多查询拆分优化，以及生成后的验证与自我纠错优化。两项优化分别对应 RAG 流程中的“召回质量”和“回答可信度”。

## 1. 优化背景

原始 RAG 流程通常是：用户提出问题，系统使用原始问题进行一次向量检索，再把检索结果作为上下文交给大模型生成回答。

这种流程实现简单，但存在两个问题：

1. 用户问题可能过短、表达模糊或关键词不完整，导致相关 chunk 没有被召回。
2. 大模型生成回答时可能没有严格依据检索上下文，出现补充不存在事实、偏离文档或幻觉的问题。

因此，本项目增加了两类优化：

1. 检索前优化：将用户问题扩展成多个查询变体，提高召回率。
2. 生成后优化：对初稿回答进行二次审查，必要时自动纠错。

## 2. 多查询拆分优化

### 2.1 实现位置

已实现文件：

```text
backend/services/search_service.py
```

核心方法：

```python
def _expand_query(self, original_query: str) -> List[str]:
```

### 2.2 实现思路

系统不再只使用用户原始 query 进行一次检索，而是在检索前调用大模型生成多个不同角度的查询变体。随后系统对每个 query 分别生成 embedding 并查询 Chroma，最后将所有结果去重、排序、截取最终 Top K。

例如，用户输入：

```text
怎么报错了？
```

可以被扩展为：

```text
系统运行时报错的原因是什么
如何排查程序异常信息
日志中错误提示对应的解决方法
```

这些查询从不同表达角度覆盖用户意图，可以提高相关文档片段的命中概率。

### 2.3 当前流程

1. 接收用户原始 query。
2. 调用 `_expand_query(query)` 生成查询列表。
3. 对每个扩展 query 创建 embedding。
4. 使用 Chroma collection 执行向量检索。
5. 使用 chunk id 对多个查询返回的结果去重。
6. 按 score 从高到低重新排序。
7. 截取最终 `top_k` 个结果返回。

### 2.4 API Key 与回退机制

查询扩展优先读取以下环境变量：

```text
ALIYUN_API_KEY
DASHSCOPE_API_KEY
```

如果存在可用 Key，会通过 DashScope OpenAI 兼容接口调用 `qwen-plus` 生成查询变体。

如果没有配置 Key，或大模型调用失败，则自动回退为：

```python
return [original_query]
```

因此，多查询优化不会破坏原有检索能力。

### 2.5 去重与排序

多查询检索可能从不同查询中命中同一个 chunk。系统使用 `seen_chunk_ids` 按 chunk id 去重，避免重复上下文进入生成阶段。

合并后的结果会统一排序：

```python
processed_results.sort(key=lambda item: item["score"], reverse=True)
processed_results = processed_results[:top_k]
```

这样最终返回的是多个查询合并后的全局 Top K。

## 3. 验证与自我纠错优化

### 3.1 实现位置

已实现文件：

```text
backend/services/generation_service.py
```

核心方法：

```python
def _critique_and_correct(
    self,
    query: str,
    context: str,
    draft_response: str,
    provider: str,
    model_name: str,
    api_key: Optional[str] = None,
) -> str:
```

### 3.2 实现思路

生成服务现在不会在初稿生成后立即返回，而是先进入审查阶段。系统会把以下三类信息交给大模型：

1. 用户原始问题。
2. 检索结果拼接出的上下文。
3. 刚生成的回答草案。

审查模型需要判断回答草案是否严格基于上下文，是否存在幻觉，是否遗漏了问题关键点。

### 3.3 当前流程

1. `generate()` 根据 provider 正常生成初稿 `response`。
2. 记录日志：`Starting Critique & Correct phase...`。
3. 调用 `_critique_and_correct()` 审查初稿。
4. 如果审查结果包含 `[PASS]`，返回原始初稿。
5. 如果审查结果包含 `[CORRECTED]`，提取修正后的回答返回。
6. 如果审查失败或没有返回明确标记，回退返回原始初稿。

### 3.4 支持范围

当前已支持自我纠错的 provider：

```text
aliyun
deepseek
```

其中：

1. `aliyun` 会复用 `_generate_with_aliyun()`。
2. `deepseek` 会复用 `_generate_with_deepseek()`，并关闭 reasoning 展示。
3. `huggingface` 当前会跳过审查，直接返回原始回答。

这样处理可以避免本地模型通道被额外调用影响稳定性。

### 3.5 返回与保存

`generate()` 中保存到 JSON 的 `response` 字段，以及接口最终返回的 `response` 字段，现在都使用审查后的 `final_response`。

也就是说：

```text
初稿生成 response -> 自我审查 -> final_response -> 保存并返回
```

## 4. 优化收益

### 4.1 检索召回率提升

多查询拆分可以缓解用户问题表达不完整的问题。相比单查询检索，它更容易覆盖同义表达、近义表达和不同问题角度，从而提升相关 chunk 被召回的概率。

### 4.2 上下文质量提升

通过 chunk id 去重和全局 score 排序，最终进入生成阶段的上下文更集中、更少重复，有助于减少无关信息对回答的干扰。

### 4.3 回答可靠性提升

自我纠错机制可以在最终返回前进行一次事实审查，降低模型脱离上下文自由发挥的概率，提高回答与文档内容的一致性。

## 5. 注意事项

1. 多查询扩展会额外调用一次大模型，因此会增加检索阶段延迟。
2. 自我纠错会在生成后额外调用一次大模型，因此会增加生成阶段延迟和 API 成本。
3. 如果没有配置 `ALIYUN_API_KEY` 或 `DASHSCOPE_API_KEY`，多查询扩展会自动回退为原始 query。
4. 如果 DeepSeek 没有配置 `DEEPSEEK_API_KEY`，DeepSeek 生成或审查会失败，并由外层逻辑记录错误。
5. 自我纠错依赖模型按约定输出 `[PASS]` 或 `[CORRECTED]`，如果模型没有输出明确标记，系统会保留原始初稿。

## 6. 后续优化建议

1. 增加开关控制是否启用多查询扩展，方便对比优化前后的效果。
2. 增加开关控制是否启用自我纠错，避免所有请求都承担额外延迟。
3. 在搜索结果中记录命中的 expanded query，方便调试和评估。
4. 将查询扩展数量设计为可配置参数，例如默认生成 3 个变体。
5. 增加评估脚本，对比优化前后的召回率、答案准确率和响应时间。

## 7. 总结

本次优化将 RAG 流程从“单次检索 + 直接生成”升级为“查询扩展 + 多路检索 + 去重排序 + 生成后审查纠错”。检索侧提升了相关上下文的召回概率，生成侧提升了最终回答的事实一致性和可靠性。

