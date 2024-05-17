class character:
    def __init__(self, name, description, gender):
        self.name = name
        self.description = description
        self.gender = gender
    
    def get_description(self):
        return self.description
    
    def get_name(self):
        return self.name
    
    def get_gender(self):
        return self.gender
    
supercat = character("Super Cat", "A black cat with green eyes and a yellow cape", "Male")
captainwhiskers = character("Captain Whiskers", "A racoon", "Male")
professorcatnip = character("Professor Catnip", "An orange cat with glasses", "Male")
ladymeowington = character("Lady Meowington", "A white cat with a pink bow", "Female")