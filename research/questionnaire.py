from config import Config
from research import gptConnector


class Questionnaire:

    def __init__(self, demo_mode=False):
        self.question_dict = {}
        self.answers_dict = {}

        if demo_mode:
            questions_link = '../data/questions_demo.txt'
            answers_link = '../data/answers_demo.txt'
        else:
            questions_link = '../data/questions.txt'
            answers_link = '../data/answers.txt'

        with open(questions_link, encoding="utf-8", errors='replace') as file:
            for index, line in enumerate(file, start=1):
                self.question_dict[
                    index] = line.strip()

        with open(answers_link, encoding="utf-8", errors='replace') as file:
            for index, line in enumerate(file, start=1):
                self.answers_dict[
                    index] = line.strip()

    def get_question(self, index: int):
        return self.question_dict.get(index)

    def get_questions(self):
        return self.question_dict

    def get_answer(self, index: int):
        return self.answers_dict.get(index)

    def get_answers(self):
        return self.answers_dict

    def set_questions(self, question_dict):
        self.question_dict.update(question_dict)

    def alternate_questions(self):
        alternated_question_dict = {}
        for key, question in self.question_dict.items():
            system = ("Create a variation of the given question. Only change tone, grammar and choice of words, "
                      "don't change the core of the question. Don't change id's neither.")
            user = f"{question}"
            alternated_question_dict.update({key: gptConnector.gpt_request(system, user)})
        self.set_questions(alternated_question_dict)


if __name__ == '__main__':
    questions = Questionnaire()
