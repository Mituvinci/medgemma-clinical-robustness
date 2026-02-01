"""
MedGemma-27B-IT Adapter

Adapter for Google's MedGemma-27B-IT (Instruction-Tuned) model using Transformers (LOCAL GPU).
Medical specialist model trained on clinical data.
"""

from typing import Dict, Any, Optional
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.agents.models.base_model import BaseLLM

logger = logging.getLogger(__name__)


class MedGemmaAdapter(BaseLLM):
    """Local GPU adapter for MedGemma-27B-IT (Instruction-Tuned) using Transformers."""

    def __init__(
        self,
        model_id: str = "google/medgemma-27b-it",
        api_key: str = None,
        device: str = "auto",
        use_4bit: bool = True,  # 4-bit quantization for local GPU (24GB)
        **kwargs
    ):
        """
        Initialize MedGemma adapter for LOCAL GPU inference.

        Args:
            model_id: Hugging Face model ID
            api_key: Hugging Face token (needed to download gated model)
            device: Device to use ("cuda", "cpu", or "auto")
            use_4bit: Use 4-bit quantization (recommended for GPUs < 80GB)
            **kwargs: Additional parameters
        """
        super().__init__(model_id, api_key, **kwargs)

        if not api_key:
            raise ValueError("HUGGINGFACE_API_KEY is required to download MedGemma")

        logger.info(f"🚀 Loading MedGemma-27B-IT locally on GPU...")
        logger.info(f"Model ID: {model_id}")
        logger.info(f"Using token: {api_key[:10]}... (length: {len(api_key)})")

        # Check GPU availability
        if torch.cuda.is_available():
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"✅ GPU detected: {gpu_name}")
            logger.info(f"   GPU memory: {gpu_memory:.2f} GB")

            # Auto-enable 4-bit if GPU has < 40GB
            if gpu_memory < 40 and not use_4bit:
                logger.warning(f"⚠️  GPU has only {gpu_memory:.1f}GB. Enabling 4-bit quantization automatically.")
                use_4bit = True
        else:
            self.device = "cpu"
            logger.warning("⚠️  No GPU detected! Using CPU (will be VERY slow)")
            use_4bit = False  # Can't use 4-bit on CPU

        # Load tokenizer
        logger.info("📥 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=api_key,
            trust_remote_code=True
        )

        # Fix tokenizer padding (critical for generation)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("   ⚠️  Set pad_token = eos_token")

        logger.info("   ✅ Tokenizer loaded")
        logger.info(f"      Vocab size: {len(self.tokenizer)}")
        logger.info(f"      PAD token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")
        logger.info(f"      EOS token: {self.tokenizer.eos_token} (ID: {self.tokenizer.eos_token_id})")

        # Configure quantization for local GPU
        quantization_config = None
        if use_4bit:
            logger.info("🔧 Enabling 4-bit quantization (saves memory, ~95-98% quality)")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            logger.info("   ✅ 4-bit config: NF4 quantization with double quantization")

        # Load model on GPU
        logger.info("📥 Loading model weights...")
        if use_4bit:
            logger.info("   (4-bit: ~14GB model + ~6GB overhead = ~20GB total)")
        else:
            logger.info("   (bfloat16: ~54GB model + ~20GB overhead = ~74GB total)")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=api_key,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16 if not use_4bit else None,
            device_map=device,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        # Get model info
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"✅ MedGemma-27B-IT loaded successfully on {self.device.upper()}!")
        logger.info(f"   Model parameters: {num_params / 1e9:.2f}B")
        if use_4bit:
            logger.info(f"   Quantization: 4-bit (memory optimized)")
        logger.info(f"   Ready for inference!")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """
        Generate text from MedGemma using LOCAL GPU.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        try:
            # Tokenize input with padding
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)

            # Generate with GREEDY decoding (more stable with quantized models)
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_tokens,
                    do_sample=False,  # Use greedy (avoids multinomial sampling bug)
                    num_beams=1,  # No beam search
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    **kwargs
                )

            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the prompt from output (return only generated text)
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()

            logger.debug(f"Generated {len(generated_text)} characters")
            return generated_text

        except Exception as e:
            logger.error(f"Error during MedGemma generation: {e}", exc_info=True)
            raise

    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output.

        Args:
            prompt: Input prompt
            schema: Expected JSON schema
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Parsed JSON object
        """
        import json
        import re

        # Add JSON formatting instructions to prompt
        json_prompt = f"""{prompt}

IMPORTANT: Respond with ONLY valid JSON. No extra text before or after.

Expected JSON format:
{json.dumps(schema, indent=2)}

Response (JSON only):"""

        try:
            response = self.generate(
                json_prompt,
                temperature=temperature,
                max_tokens=2048,
                **kwargs
            )

            # Extract JSON from response (handle cases with markdown code blocks)
            json_str = self._extract_json(response)

            # Parse JSON
            parsed = json.loads(json_str)

            logger.debug("Successfully parsed structured output")
            return parsed

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from MedGemma: {e}")
            logger.debug(f"Raw response: {response}")
            raise ValueError(f"Invalid JSON response from MedGemma: {e}")

    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from text (handles markdown code blocks).

        Args:
            text: Text potentially containing JSON

        Returns:
            Clean JSON string
        """
        import re

        # Remove markdown code blocks if present
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)

        # Try to find JSON object
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0)

        # Try to find JSON array
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return match.group(0)

        # Return as-is if no patterns found
        return text.strip()

    def count_tokens(self, text: str) -> int:
        """
        Count tokens using actual tokenizer.

        Args:
            text: Input text

        Returns:
            Actual token count
        """
        return len(self.tokenizer.encode(text))

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "local_gpu"

    def generate_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """
        Generate with system and user prompts.

        MedGemma uses special formatting for instructions.

        Args:
            system_prompt: System instructions
            user_prompt: User message
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        # Format for MedGemma instruction-tuned model
        formatted_prompt = f"""<|system|>
{system_prompt}
<|end|>

<|user|>
{user_prompt}
<|end|>

<|assistant|>"""

        return self.generate(formatted_prompt, temperature, max_tokens, **kwargs)
