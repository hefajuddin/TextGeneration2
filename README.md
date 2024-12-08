# TextGeneration2

1. Install requirment using following commands-
pip install -r requirements.txt

2. Add different context or articles in data/context_data.json files where from you want to get text-generation

3. Run cmd on your project directory and execute the project using following command-
python generate.py

4. Server will run on http://127.0.0.1:5004/, ensure that this port is not runnig in another instance

5. Go to browser, hit http://127.0.0.1:5004/ now UI is displaying

6. Input any question on your context given in context_data.json and hit enter, then you will get answer

=============How the project works===============
1. When user input his/her question or statement, retrieve_context() read the question and extract possible contexts and return
2. Then load_generation_model() load the model gpt2
3. generate_text() generates an answer or statement on your query from relevant context
