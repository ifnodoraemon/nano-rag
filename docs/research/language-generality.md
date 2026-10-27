# 英语端到端：找到两处硬编码语言假设

上一份文档记录了检索层的跨语言问题（查询扩展的字符上限）。这份记录**生成层**的问题——
它只有在跑完整管道、看模型真正输出的内容时才会暴露。

## 现象

用 196 条英语提问跑美国法典语料，系统回答：

> **"文档中未包含关于'written claim requirements for substitute check disputes'的相关信息。"**

用户用英语提问，系统用**中文**拒答。

## 两个根因，都在代码里

### ① 提示词里的硬编码拒答模板

```python
# app/generation/prompt_builder.py
【防幻觉与精简指令】：……请直接回答："文档中未包含关于【…】的相关信息。"
```

整个提示词套件都是中文写的，并且**指定了拒答时的确切句式**。模型遵循了这条指令。

### ② 更隐蔽的：拒答文案由代码拼装

```python
# app/agentic/service.py
if not is_answerable:
    ans = f"文档中未包含关于"{'、'.join(missing)}"的相关信息。"
```

**提示词里的语言规则根本管不到这里**——这句话是 f-string 生成的，不是模型写的。
这是关键：加多少条"用问题语言回答"的提示都没有用。

## 修法

**提示词侧**：删除硬编码句式，并加入一条通用语言契约

```
LANGUAGE (highest priority): write the ENTIRE reply in the language the user's question
is written in, even when the surrounding instructions above are in a different language.
This applies to every part of the reply - the answer, an insufficiency notice, a refusal,
and any explanation.
```

这条契约**不假设任何具体语言**，只要求"跟随提问语言"。

**代码侧**：拒答文案改由模型提供

```python
stated = parsed.get("extracted_answer")
if not stated:
    raise ModelOutputError("an unanswerable verdict must carry its own explanation")
```

`missing_entities` 保留为结构化信号（拒答率等指标从它读取），但**用户可见的那句话由模型写**。

## 效果

| 指标 | 修复前 | 修复后 |
| --- | --- | --- |
| 报错 | 0 | 0 |
| **回答语言** | 中英混杂（拒答为中文） | **英文 40/40** |
| 延迟中位 | 9.1s | 9.4s |
| `citation_integrity` | 0.8409 | **0.8889** |

中文侧回归：正常回答与拒答均为中文，未见退化。

## 为什么这个 bug 只能靠多语言测试发现

- 中文语料上**永远测不出来**：系统用中文拒答，看起来完全正常
- 单元测试也测不出来：`_render_structured_answer` 的旧测试断言 `"术语X" in answer`，
  用中文输入、中文断言，自洽地通过
- 只有在**用英语提问、检查回答语言**时才会暴露

这与上一份文档中查询扩展的字符上限是同一类问题：**一个只在单一语言下验证过的系统，
会把"中文的具体值"误当作"通用的规则"。**

## 一个过程中的自我纠错

我一度以为中文本轮出现了回归（拒答变多）。检查发现是我把中文查询跑在了
**美国法典语料**上——那里当然没有中文法规。用正确语料重测，中文表现正常。

**测试用错语料会产生假的回归信号**，和用错指标一样危险。
