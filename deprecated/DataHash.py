import hashlib

def DataHash(data):
    # convert data to bytes
    if isinstance(data, str):
        data = data.encode()
    elif isinstance(data, int) or isinstance(data, float):
        data = str(data).encode()
    
    # compute the hash
    md5_hash = hashlib.md5()
    md5_hash.update(data)
    return md5_hash.hexdigest()

# test
if __name__ == "__main__":
    hash_val = DataHash("测试数据")
    print(hash_val)
