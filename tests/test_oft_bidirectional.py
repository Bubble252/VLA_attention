"""Run only in pinned OFT fork; verifies actual attention semantics on CPU."""
import pytest
torch=pytest.importorskip('torch')
from transformers import LlamaConfig,LlamaModel


def make_model(implementation):
    config=LlamaConfig(vocab_size=64,hidden_size=16,intermediate_size=32,
                       num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,
                       max_position_embeddings=16,attention_dropout=0.)
    config._attn_implementation=implementation
    torch.manual_seed(17)
    return LlamaModel(config).eval()


def test_oft_sdpa_observes_future_tokens_but_masks_padding():
    model=make_model('sdpa')
    a=torch.tensor([[1,2,3,4]]); b=torch.tensor([[1,2,3,5]])
    with torch.no_grad():
        full=model(a,use_cache=False).last_hidden_state[:,0]
        changed=model(b,use_cache=False).last_hidden_state[:,0]
        assert not torch.allclose(full,changed,atol=1e-6), 'This is not the bidirectional OFT fork'
        padding=torch.tensor([[1,1,1,0]])
        masked_a=model(a,attention_mask=padding,use_cache=False).last_hidden_state[:,0]
        masked_b=model(b,attention_mask=padding,use_cache=False).last_hidden_state[:,0]
        torch.testing.assert_close(masked_a,masked_b)
    assert not torch.cuda.is_initialized()


def test_eager_path_is_not_an_equivalent_oft_fallback():
    model=make_model('eager')
    with torch.no_grad():
        a=model(torch.tensor([[1,2,3,4]]),use_cache=False).last_hidden_state[:,0]
        b=model(torch.tensor([[1,2,3,5]]),use_cache=False).last_hidden_state[:,0]
    torch.testing.assert_close(a,b)
