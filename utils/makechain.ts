import { OpenAIChat } from 'langchain/llms';
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from 'langchain/chains';
import { PineconeStore } from 'langchain/vectorstores';
import { PromptTemplate } from 'langchain/prompts';
import { CallbackManager } from 'langchain/callbacks';

const CONDENSE_PROMPT =
  PromptTemplate.fromTemplate(`Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`);

const QA_PROMPT = PromptTemplate.fromTemplate(
  `You are an AI assistant tasked with retrieving information from a data sheet for the options and pricing of various vehicles and accessories available for purchase under a specific Florida State Term Contract. The contract number and the name of the vendor can be found near the beginning of the document. The document contains line items of available makes and models from many major manufacturers. For example "Chevy" is a manufacturer, this is also known as a Make. "Silverado" is a Model, and "1500 Crew Cab" refers to the specific Package for this Make and Model combination. Other Make, Model and Package combinations include "FORD" "F-150" and "SUPER CREW CAB XL 157''WB SSV 4WD", or "TOYOTA" "SIENNA" and "FWD LE HYBRID 8-PASSENGER".  You will use your knowledge to successfully identify all of the other Make, Model and Packages available in this contract. There are additionally various addons and accessories listed in the data, including items under categories such as "Seat Upgrade" "Floor Protection" "Hitches & Accessories" and others. The data is formatted into columns, with headings for Year, Model Code, Model Description, MSRP and Contract Price. You are given the following extracted parts of a long document and a question. Provide a conversational answer based on the context provided. You must NEVER format responses into a table, only use lists. Generally speaking answers should be formatted as an easily readable list and include details of Year, Make, Model, Package, MSRP price and Contract Price, but omit the Model Code. Also reference the contract #, the PDF file name and any relevant page numbers in which the referenced results can be found. You should only provide hyperlinks that reference the context below. Do NOT make up hyperlinks. If you can't find the answer in the context below, just say "Hmm, I'm not sure." Don't try to make up an answer. If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

Question: {question}
=========
{context}
=========
Answer in Markdown:`,
);

export const makeChain = (
  vectorstore: PineconeStore,
  onTokenStream?: (token: string) => void,
) => {
  const questionGenerator = new LLMChain({
    llm: new OpenAIChat({ temperature: 0 }),
    prompt: CONDENSE_PROMPT,
  });
  const docChain = loadQAChain(
    new OpenAIChat({
      temperature: 0,
      modelName: 'gpt-3.5-turbo', //change this to older versions (e.g. gpt-3.5-turbo) if you don't have access to gpt-4
      streaming: Boolean(onTokenStream),
      callbackManager: onTokenStream
        ? CallbackManager.fromHandlers({
            async handleLLMNewToken(token) {
              onTokenStream(token);
              console.log(token);
            },
          })
        : undefined,
    }),
    { prompt: QA_PROMPT },
  );

  return new ChatVectorDBQAChain({
    vectorstore,
    combineDocumentsChain: docChain,
    questionGeneratorChain: questionGenerator,
    returnSourceDocuments: true,
    k: 2, //number of source documents to return
  });
};
