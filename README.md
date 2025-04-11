# Train the model
- python pet-adoption-model-improved.py
# Make single predictions
- python pet-adoption-prediction.py single --type Dog --age 3 --breed "Mixed Breed" --gender Male --color1 Brown --color2 White --size Medium --fur Medium --vaccinated Yes --sterilized No --health Healthy --fee 0 --photos 7 --description "Friendly dog who loves to play fetch and is good with children."
# Process batch predictions
- python pet-adoption-prediction.py batch --input new_pets.csv --output predictions.csv