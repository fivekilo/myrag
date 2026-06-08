import os
from pathlib import Path
import dotenv

dotenv.load_dotenv(Path(__file__).resolve().parent.parent / ".env")
import json
from datetime import datetime
from typing import List, Dict, Optional
import logging
import time
import requests
from utils.model_utils import get_huggingface_model_path
from utils.paths import GENERATION_RESULTS_DIR, workspace_relative

try:
    import torch
except Exception:  # pragma: no cover - optional dependency at runtime
    torch = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency at runtime
    OpenAI = None

try:
    from langchain_core.prompts import ChatPromptTemplate
except Exception:  # pragma: no cover - optional dependency at runtime
    ChatPromptTemplate = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except Exception:  # pragma: no cover - optional dependency at runtime
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None

try:
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
except Exception:  # pragma: no cover - optional dependency at runtime
    ChatHuggingFace = None
    HuggingFacePipeline = None

# 设置环境变量以启用 Apple Silicon (MPS) 回退到 CPU (当遇到不支持的操作时会自动回退到 CPU 执行)
# 目前 PyTorch 版本 ≥ 1.13 时，才支持 Apple 的 Metal Performance Shaders (MPS) ，而且暂不支持「多 GPU」，另外，部分训练操作尚未完全实现
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

logger = logging.getLogger(__name__)


class GenerationService:
    """
    生成服务类：负责调用不同的模型提供商（HuggingFace、OpenAI、DeepSeek）生成回答
    支持本地模型和API调用，并将生成结果保存到文件
    """

    def __init__(self):
        """
        初始化生成服务，配置支持的模型列表和创建输出目录
        """
        self.model = ""
        self.tokenizer = ""
        self.history = ""
        self.models = {
            "huggingface": {
                "Llama-2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
                "DeepSeek-7b": "deepseek-ai/deepseek-llm-7b-chat",
                "DeepSeek-R1-Distill-Qwen": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                "Qwen-Qwen3-1.7B": "Qwen/Qwen3-1.7B",
            },
            "aliyun": {
                "qwen-turbo": "qwen-turbo",
                "qwen-plus": "qwen-plus",
                "qwen3.6-plus": "qwen3.6-plus",
                "qwen3.7-max": "qwen3.7-max",
                "qwen3.7-max-2026-05-20": "qwen3.7-max-2026-05-20",
            },
            "deepseek": {
                "deepseek-v4-flash": "deepseek-v4-flash",
                "deepseek-v4-pro": "deepseek-v4-pro",
                "deepseek-chat": "deepseek-chat",
                "deepseek-reasoner": "deepseek-reasoner",
            },
        }

        # 确保输出目录存在
        GENERATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def _load_huggingface_model(self, model_name: str):
        """
        加载HuggingFace模型

        参数:
            model_name: 模型名称，对应self.models["huggingface"]中的键

        返回:
            model: 加载的模型
            tokenizer: 对应的分词器
        """
        try:
            if (
                AutoModelForCausalLM is None
                or AutoTokenizer is None
                or pipeline is None
                or ChatHuggingFace is None
                or HuggingFacePipeline is None
                or torch is None
                or ChatPromptTemplate is None
            ):
                raise ImportError(
                    "HuggingFace generation dependencies are not installed. "
                    "Install torch, transformers, langchain_core, and langchain_huggingface to use local models."
                )

            tensor_device = "cuda" if torch.cuda.is_available() else "cpu"
            model_name = self.models["huggingface"][model_name]
            model_name = get_huggingface_model_path(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=tensor_device,  # 自动分配GPU/CPU
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )

            text_gen_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=1024,  # 控制生成文本的最大长度
                temperature=0.7,  # 控制生成随机性
                top_k=50,  # 限制候选词数量
                top_p=0.9,  # 核采样参数
                num_return_sequences=1,  # 生成单个序列
                truncation=True,  # 启用输入截断
                pad_token_id=tokenizer.eos_token_id,  # 填充token设置
                clean_up_tokenization_spaces=False,  # 保留原始分词空格
            )

            hf_pipeline = HuggingFacePipeline(pipeline=text_gen_pipeline)
            chat_model = ChatHuggingFace(llm=hf_pipeline)

            return chat_model, tokenizer
        except Exception as e:
            logger.error(f"Error loading HuggingFace model: {str(e)}")
            raise

    def _generate_with_huggingface(
        self,
        model_name: str,
        query: str,
        context: str,
        load_model: bool,
        max_length: int = 1024,
    ) -> str:
        """
        使用HuggingFace模型生成回答

        参数:
            model_name: 模型名称
            query: 用户查询
            context: 上下文信息
            max_length: 生成文本的最大长度

        返回:
            生成的回答文本
        """
        try:
            if bool(load_model) == True:
                self.model, self.tokenizer = self._load_huggingface_model(model_name)

            # 构建提示
            # prompt = f"""请基于以下上下文回答问题。如果上下文中没有相关信息，请说明无法回答。
            prompt = ChatPromptTemplate.from_template(
                """请基于上下文与回话记录回答问题。如果上下文和回话记录中没有相关信息，请直接根据问题回答。
                        回话记录：{history}
                        问题：{query}

                        上下文：
                        {context}

                        回答："""
            )

            # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            # outputs = model.generate(
            #     **inputs,
            #     max_length=max_length,
            #     num_return_sequences=1,
            #     temperature=0.7,
            #     do_sample=True
            # )

            # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            ts = time.time()
            answer = self.model.invoke(
                prompt.format(query=query, history=self.history, context=context)
            )
            spent_sec = int(time.time() - ts)
            text = answer.content
            text_parts = text.split(r"<think>")
            parts = text_parts[1].split(r"</think>")
            thinkingInfo = parts[0]
            responseInfo = parts[1]

            self.history += f"""
用户提问：{query}
AI回复：{responseInfo}

"""
            answer_content = f"""s:{spent_sec}
用户提问：{query}
AI思考过程：{thinkingInfo}
AI回复：{responseInfo}

"""
            # return answer.split("回答：")[-1].strip()
            return answer_content

        except Exception as e:
            logger.error(f"Error generating with HuggingFace: {str(e)}")
            raise

    def _generate_with_aliyun(
        self,
        model_name: str,
        query: str,
        context: str,
        api_key: Optional[str] = None,
        show_reasoning: bool = True,
    ) -> str:
        """
        使用OpenAI API生成回答

        参数:
            model_name: 模型名称
            query: 用户查询
            context: 上下文信息
            api_key: OpenAI API密钥，如不提供则从环境变量获取

        返回:
            生成的回答文本
        """
        try:
            if OpenAI is None:
                raise ImportError(
                    "openai package is not installed. Install it to call Aliyun Bailian via compatible-mode."
                )

            resolved_api_key = api_key or os.getenv("ALIYUN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
            if not resolved_api_key:
                raise ValueError("Aliyun Bailian API key not provided")

            base_url = os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

            # 初始化OpenAI客户端
            client = OpenAI(
                api_key=resolved_api_key,
                base_url=base_url,
            )

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Use the provided context to answer the question.",
                },
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"},
            ]

            extra_body = {"enable_thinking": bool(show_reasoning)}
            completion = client.chat.completions.create(
                model=self.models["aliyun"][model_name],
                messages=messages,
                # 百炼官方文档当前仍建议通过 extra_body 传入 enable_thinking。
                extra_body=extra_body,
                stream=True,
                stream_options={"include_usage": True},
            )
            reasoning_content = ""  # 完整思考过程
            answer_content = ""  # 完整回复
            is_answering = False  # 是否进入回复阶段
            if show_reasoning:
                print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

            for chunk in completion:
                if not chunk.choices:
                    print("\n" + "=" * 20 + "Token 消耗" + "=" * 20 + "\n")
                    print(chunk.usage)
                    continue

                delta = chunk.choices[0].delta

                # 只收集思考内容
                if (
                    hasattr(delta, "reasoning_content")
                    and delta.reasoning_content is not None
                ):
                    if show_reasoning and not is_answering:
                        print(delta.reasoning_content, end="", flush=True)
                    reasoning_content += delta.reasoning_content

                # 收到content，开始进行回复
                if hasattr(delta, "content") and delta.content:
                    if not is_answering:
                        print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                        is_answering = True
                    print(delta.content, end="", flush=True)
                    answer_content += delta.content
            final_answer = answer_content.strip()
            if show_reasoning and reasoning_content.strip():
                return f"【思维过程】\n{reasoning_content.strip()}\n\n【最终答案】\n{final_answer}"
            return final_answer

        except Exception as e:
            logger.error(f"Error generating with Aliyun Bailian: {str(e)}")
            raise

    def _generate_with_deepseek(
        self,
        model_name: str,
        query: str,
        context: str,
        api_key: Optional[str] = None,
        show_reasoning: bool = True,
    ) -> str:
        """
        使用DeepSeek API生成回答

        参数:
            model_name: 模型名称
            query: 用户查询
            context: 上下文信息
            api_key: DeepSeek API密钥，如不提供则从环境变量获取
            show_reasoning: 是否显示推理过程（仅对推理模型有效）

        返回:
            生成的回答文本，对于推理模型可能包含思维过程
        """
        try:
            if OpenAI is None:
                raise ImportError("openai package is not installed. Install it to call DeepSeek APIs.")

            if not api_key:
                api_key = os.getenv("DEEPSEEK_API_KEY")
                if not api_key:
                    raise ValueError("DeepSeek API key not provided")

            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Use the provided context to answer the question.",
                },
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"},
            ]

            response = client.chat.completions.create(
                model=self.models["deepseek"][model_name],
                messages=messages,
                max_tokens=512,
                stream=False,
                reasoning_effort=(
                    "high"
                    if show_reasoning
                    and model_name in {"deepseek-v4-flash", "deepseek-v4-pro"}
                    else None
                ),
                extra_body=(
                    {"thinking": {"type": "enabled" if show_reasoning else "disabled"}}
                    if model_name in {"deepseek-v4-flash", "deepseek-v4-pro"}
                    else None
                ),
            )

            message = response.choices[0].message
            reasoning = getattr(message, "reasoning_content", None)
            answer = (message.content or "").strip()

            if show_reasoning and reasoning:
                return f"【思维过程】\n{reasoning}\n\n【最终答案】\n{answer}"

            return answer

        except Exception as e:
            logger.error(f"Error generating with DeepSeek: {str(e)}")
            raise

    def _critique_and_correct(
        self,
        query: str,
        context: str,
        draft_response: str,
        provider: str,
        model_name: str,
        api_key: Optional[str] = None,
    ) -> str:
        """Check whether the draft is grounded in context, and rewrite it if needed."""
        prompt = f"""你是一个严格的纠错审查员。
请检查【回答草案】是否准确且严格地基于给定的【上下文】来回答了【问题】。

【问题】: {query}
【上下文】: {context}
【回答草案】: {draft_response}

评判规则：
1. 如果草案里捏造了上下文没有的数据（幻觉），或者未回答问题，请输出：[CORRECTED] 加上基于上下文重写后的严谨回答。
2. 如果草案本身已经很好基于了上下文，并且准确无误，请只输出：[PASS]。
"""

        try:
            if provider == "aliyun":
                critique = self._generate_with_aliyun(
                    model_name, prompt, "[NO CONTEXT NEEDED]", api_key, show_reasoning=False
                )
            elif provider == "deepseek":
                critique = self._generate_with_deepseek(
                    model_name,
                    prompt,
                    "[NO CONTEXT NEEDED]",
                    api_key,
                    show_reasoning=False,
                )
            else:
                logger.info(f"Critique skipped for unsupported provider: {provider}")
                return draft_response

            if "[PASS]" in critique:
                logger.info("Critique passed. Using original draft.")
                return draft_response

            if "[CORRECTED]" in critique:
                logger.info("Critique found issues. Using corrected response.")
                return critique.split("[CORRECTED]", 1)[-1].strip()

            logger.info("Critique returned no clear marker. Using original draft.")
            return draft_response
        except Exception as e:
            logger.error(f"Correction failed: {e}")
            return draft_response

    def generate(
        self,
        provider: str,
        model_name: str,
        query: str,
        search_results: List[Dict],
        load_model: bool,
        api_key: Optional[str] = None,
        show_reasoning: bool = True,
    ) -> Dict:
        """
        生成回答并保存结果

        参数:
            provider: 模型提供商，可选值为"huggingface"、"openai"、"deepseek"
            model_name: 模型名称
            query: 用户查询
            search_results: 搜索结果列表，用于构建上下文
            api_key: API密钥（对于API调用）
            show_reasoning: 是否显示推理过程（仅对DeepSeek推理模型有效）
            load_model: 是否装载模型

        返回:
            包含生成回答和保存路径的字典
        """
        try:
            # 准备上下文
            context = "\n\n".join(
                [
                    f"[Source {i + 1}]: {result['text']}"
                    for i, result in enumerate(search_results)
                ]
            )

            ts = time.time()
            # 根据不同提供商生成回答
            if provider == "huggingface":
                response = self._generate_with_huggingface(
                    model_name, query, context, load_model
                )
            elif provider == "aliyun":
                response = self._generate_with_aliyun(
                    model_name, query, context, api_key, show_reasoning
                )
            elif provider == "deepseek":
                response = self._generate_with_deepseek(
                    model_name, query, context, api_key, show_reasoning
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            logger.info("Starting Critique & Correct phase...")
            final_response = self._critique_and_correct(
                query, context, response, provider, model_name, api_key
            )

            # 准备保存的结果
            result = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "provider": provider,
                "model": model_name,
                "response": final_response,
                "context": search_results,
            }

            # 生成文件名并保存
            spend_sec = int(time.time() - ts)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"generation_{provider}_{model_name}_{spend_sec}s_{timestamp}_{load_model}.json"
            filepath = GENERATION_RESULTS_DIR / filename

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            return {
                "response": final_response,
                "saved_filepath": workspace_relative(filepath),
            }

        except Exception as e:
            logger.error(f"Error in generation: {str(e)}")
            raise

    def get_available_models(self) -> Dict:
        """
        获取可用的模型列表

        返回:
            包含所有支持模型的字典
        """
        return self.models
