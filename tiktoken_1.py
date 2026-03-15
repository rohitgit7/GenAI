import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")
encoding = tiktoken.encoding_for_model("gpt-5-chat")
encoding.encode("tiktokenisgreat!")

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    print("Token List:",encoding.encode(string))
    num_tokens = len(encoding.encode(string))
    return num_tokens

print("Token Count:",num_tokens_from_string("tiktoken is great! h", "o200k_base"))