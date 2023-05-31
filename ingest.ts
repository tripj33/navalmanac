import { HNSWLib } from "langchain/vectorstores";
import { OpenAIEmbeddings } from "langchain/embeddings";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { TextLoader } from "langchain/document_loaders";

const FILENAME = "Pathophysiology.txt";

export const run = async () => {
  const loader = new TextLoader(FILENAME);
  const rawDocs = await loader.load();
  console.log("Loader created.");
  /* Split the text into chunks */
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const docs = await textSplitter.splitDocuments(rawDocs);
  console.log("Docs splitted.");

  console.log("Creating vector store...");
  /* Create the vectorstore */
  const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings({openAIApiKey:"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"}));
  await vectorStore.save("data");
};

(async () => {
  await run();
  console.log("done");
})();
