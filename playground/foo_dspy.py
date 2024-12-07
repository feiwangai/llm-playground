import dspy


def explore_fun1():
    lm = dspy.LM("openai/gpt-4o")
    dspy.configure(lm=lm)
    qa = dspy.Predict("question: str -> response: str")
    print(qa(question="What are high memory and low memory on Linux?").response)
    print(dspy.inspect_history(n=1))


def explore_fun2():
    lm = dspy.LM("openai/gpt-4o")
    dspy.configure(lm=lm)
    cot = dspy.ChainOfThought("question -> response")
    print(cot(question="should curly braces appear on their own line?").response)
    print(dspy.inspect_history(n=1))


if __name__ == "__main__":
    # explore_fun1()
    explore_fun2()
