from transformers import AutoTokenizer
import torch

class TokenDecoder:
    """Token解码器"""
    
    def __init__(self, model_name=None, tokenizer=None):
        """初始化解码器"""
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif model_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            raise ValueError("需要提供model_name或tokenizer")
    
    def decode_tokens(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        """
        将token IDs解码为文本
        
        Args:
            token_ids: token IDs列表或张量
            skip_special_tokens: 是否跳过特殊token
            clean_up_tokenization_spaces: 是否清理空格
        
        Returns:
            解码后的文本
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
        
        return self.tokenizer.decode(token_ids, 
                                     skip_special_tokens=skip_special_tokens,
                                     clean_up_tokenization_spaces=clean_up_tokenization_spaces)
    
    def decode_batch(self, batch_token_ids, **kwargs):
        """批量解码"""
        decoded_texts = []
        for token_ids in batch_token_ids:
            decoded = self.decode_tokens(token_ids, **kwargs)
            decoded_texts.append(decoded)
        return decoded_texts
    
    def decode_with_details(self, token_ids):
        """带详细信息的解码"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
        
        print(f"Token数量: {len(token_ids)}")
        print(f"Token IDs: {token_ids}")
        
        # 逐个token解码
        print("\n逐个Token解码:")
        for i, token_id in enumerate(token_ids):
            token_text = self.tokenizer.decode([token_id])
            print(f"  [{i:3d}] ID: {token_id:6d} -> '{token_text}'")
        
        # 整体解码
        full_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"\n整体解码: {full_text}")
        
        return full_text

# 使用示例
if __name__ == "__main__":
    model_path = "/home/featurize/work/InfiniFamily/cache/models--inclusionAI--LLaDA-MoE-7B-A1B-Instruct/snapshots/783d3467f108d28ac0a78d3e41af16ab05cabd8d"
    # 示例1: 使用模型名称
    decoder = TokenDecoder(model_path)
    
    # 示例token IDs (假设的)
    token_ids = [157151,  90827, 157152,   2496,    449,    259,   9031,  12458,  16841,
             13,    198,  14136,   5381,   6350,    928, 156900, 157151,     39,
         116171, 157152,  86059,    560,   2001,    220,     16,     17,  44137,
            854,   6984,    352,    220,     19,   3871,     13,   4474,    378,
             11,   1285,   9660,    220,     21,  44137,    854,   6984,     13,
           2071,   1494,  44137,    560,   1285,   2001,    296,    220,     23,
           3871,     30, 156900, 157151,   8469,   7342,   5468, 157152]
    
    # 解码
    text = decoder.decode_tokens(token_ids)
    print(f"解码结果: {text}")
    
    # 详细解码
    decoder.decode_with_details(token_ids)