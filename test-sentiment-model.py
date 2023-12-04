from transformers import pipeline, MarianTokenizer, MarianMTModel

# Get the name of the model
model_name = 'Helsinki-NLP/opus-mt-en-fr'

# Get the tokenizer
tokenizer = MarianTokenizer.from_pretrained(model_name)
# Instantiate the model
model = MarianMTModel.from_pretrained(model_name)

def format_batch_texts(language_code, batch_texts):
    formated_bach = [">>{}<< {}".format(language_code, text) for text in batch_texts]
    return formated_bach

def perform_translation(batch_texts, model, tokenizer, language="fr"):
    # Prepare the text data into appropriate format for the model
    formated_batch_texts = format_batch_texts(language, batch_texts)

    # Generate translation using model
    translated = model.generate(**tokenizer(formated_batch_texts, return_tensors="pt", padding=True))

    # Convert the generated tokens indices back into text
    translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    return translated_texts

if __name__ == "__main__":
    english_texts = input("Jeffrey please input:")
    
    print(f"The texts is: {english_texts}")

    model_checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

    distil_bert_model = pipeline(task="sentiment-analysis", model=model_checkpoint)

    print(distil_bert_model(english_texts[1:]))

    '''
    # Check the model translation from the original language (English) to French
    translated_texts = perform_translation(english_texts, model, tokenizer)

    # Create wrapper to properly format the text
    from textwrap import TextWrapper
    # Wrap text to 80 characters
    wrapper = TextWrapper(width=80)

    for text in translated_texts:
        print("Original text: \n", text)
        print("Translation : \n", text)
        print(print(wrapper.fill(text)))
        print("")
    '''
