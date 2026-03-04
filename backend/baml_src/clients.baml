// Chart Eye — Azure OpenAI GPT-4.1 vision client
// Endpoint: https://advance.openai.azure.com
// Deployment: gpt-4.1-scrape

client<llm> AzureGPT41 {
  provider azure-openai
  options {
    model "gpt-4.1-scrape"
    api_key env.AZURE_OPENAI_KEY
    base_url "https://advance.openai.azure.com/openai/deployments/gpt-4.1-scrape"
    api_version "2025-01-01-preview"
  }
}
