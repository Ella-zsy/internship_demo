from rag.rag_chain import rag_chain


def main():

    while True:

        question = input("\nQuestion: ")

        if question == "exit":
            break

        answer = rag_chain.invoke(question)

        print("\nAnswer:", answer)


if __name__ == "__main__":
    main()