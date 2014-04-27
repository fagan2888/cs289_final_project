def count_unique_values(iterable):
    temp = set()
    result = list()
    for x in iterable:
        temp.add(x)
    return len(temp)
