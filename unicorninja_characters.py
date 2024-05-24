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

sparkle = character("Sparkle", "A fierce Unicorn with a shimmering rainbow mane", "Female")
shadow = character("Shadow", "A stealthy black Ninja with a mysterious past", "Male")
luna = character("Luna", "A mystical Unicorn with a silvery coat and moonlit eyes", "Female")
blade = character("Blade", "A skilled Ninja warrior with unmatched agility", "Male")
storm = character("Storm", "A powerful Unicorn with a thunderous presence and electric blue horn", "Female")
ember = character("Ember", "A fiery Ninja with a passion for justice and red-tinted armor", "Male")
