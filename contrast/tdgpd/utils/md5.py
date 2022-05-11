from hashlib import md5


def get_md5(filename):
    hash_md5 = md5()
    with open(filename, "rb") as f:
        hash_md5.update(f.read())
    return hash_md5.hexdigest()


if __name__ == "__main__":
    md5 = get_md5(__file__)
    print("MD5 of {} is {}".format(__file__, md5))
