from setuptools import find_packages, setup 
from typing import List 


HYPHEN_E_DOT_REQUIREMENT = '-e .'

# Create a function to read and return list of requirements 
def get_requirements(file_path:str) -> List[str]:
    # Create an empty list 
    requirements = []

    # Open the requirements file 
    with open(file_path) as file_obj:
        # Read each line 
        for line in file_obj:
            # Remove the newline character at end of each line 
            requirements.append(line.replace("\n", ""))

    # Remove the '-e .' requirement 
    if HYPHEN_E_DOT_REQUIREMENT in requirements:
        requirements.remove(HYPHEN_E_DOT_REQUIREMENT)


    return requirements

setup(
    name='Predicting Customer Lifetime Value',
    version='0.1.0',
    packages=find_packages(),
    author="Western Onzere",
    author_email="minichworks@gmail.com",
    url="https://github.com/minich-code/clv/tree/main",
    description="Predicting Customer Lifetime Value",
    #long_description=open('README.md').read(),
    install_requires=get_requirements('requirements.txt')
    
)
