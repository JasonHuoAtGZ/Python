

class Mother():
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def print_info(self):
        print(self.name, self.age)


m1=Mother("rose", 50)
m1.print_info()


class Son(Mother):
    def __init__(self, name, age):
        super().__init__(name=name, age=age)

    def print_info_son(self):
        self.print_info()
        print("'s son")

    def son_fun(self):
        s1 = Son("Shirley", 40)
        s1.print_info_son()

s1 = Son("Laura", 30)

s1.print_info_son()
s1.son_fun()
