from concurrent.futures import ThreadPoolExecutor

values = [2, 3, 4, 5]


def task(n):
    return n * n


def main():
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = executor.map(task, values)
    for result in results:
        print(result)


if __name__ == '__main__':
    main()
