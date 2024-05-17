class character:
    def __init__(self, name, description, gender):
        self.name = name
        self.description = description
        self.gender = gender

    def __repr__(self):
        return f"{self.name} is {self.description}."
    
    def get_description(self):
        return self.description
    
    def get_name(self):
        return self.name
    
    def get_gender(self):
        return self.gender
    
tina = character("Tina", "A green Tyrannosaurus Rex", "Female")
trixie = character("Trixie", "An orange Triceratops", "Female")
vicky = character("Vicky", "A green velocoraptor", "Female")
benny = character("Benny", "A grey Brachiosaurus", "Male")