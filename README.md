# InspireMe
A program leveraging machine learning to turn your frown upside down by taking statements 
from you and responding with inspirational statements (or the opposite if you seem too happy).

### Initial Plan
1. Allow users to input a sentence about how they're feeling.
   - `I feel shitty today.`
2. Utilize an RNN to classify the text into a category related to their feelings 
   - `I feel shitty today.` -> `sadness`
3. Use the class and the text itself to relate a generated quote to the given text 
    - `“There's always tomorrow and it always gets better.” - Ariana Grande`

### Datasets Being Used
- [Text to Emotion Classification](https://huggingface.co/datasets/dair-ai/emotion)
- [Emotion to Quote Generation](https://www.kaggle.com/datasets/manann/quotes-500k)
