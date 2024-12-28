# Stock Recommendation AI Agent Team

phiData provides a sophisticated framework that ensures data integrity and enhances the capabilities of agents to interact and process complex datasets seamlessly.

**Overview**

This project leverages a sophisticated multi-agent orchestration system to provide comprehensive stock market analysis and recommendations. It integrates advanced data fetching, analysis, and visualization tools to deliver real-time insights and stock performance predictions.

**Key Features**
- **Multi-Agent Orchestration:** Utilizes multiple specialized agents that coordinate to handle complex queries about stock market data, news, and analytics.
- **Dynamic Stock Market Analysis:** Offers deep dives into stock fundamentals, analyst recommendations, and market sentiment analysis.
- **Interactive Query Handling:** Processes user queries effectively, providing tailored responses based on real-time data.

![image](https://github.com/wasimhassanshah/Stock_Recommendation_AI_Agent_Team/blob/main/Stock_agent.PNG)

![image](https://github.com/wasimhassanshah/Stock_Recommendation_AI_Agent_Team/blob/main/multi_agent_team_stock_recommendation.PNG)


**Technology Stack** 
- Groq: Main model powering each agent, utilizing Groq's llama-3.1-8b-instant for instant responses with high accuracy.
- YFinanceTools: For fetching live stock prices, fundamentals, and analyst recommendations.
- DuckDuckGo: Used by the News Agent to fetch and summarize relevant news articles.
- Dotenv: Manages environment variables securely.
- Matplotlib & Pandas: Essential for data manipulation and generating visualizations, such as stock price trends.
- phiData: Part of the phi.agent library, used for data framing and structuring in the agents to ensure data consistency and effective processing.

# Architecture
**Agents**
- Research Agent: Specializes in tabular data display and financial metrics extraction using phiData to frame and structure output.
- Data Analysis Agent: Focuses on in-depth analysis of stock fundamentals, leveraging phiData for coherent data presentation.
- Plotting Agent: Generates and saves stock price plots, enhancing data presentation with structured data from phiData.
- Query Agent: Acts as the first line of interaction, parsing and directing user queries, with phiData ensuring query data is accurately framed.
- Recommendation Agent: Provides buying or selling recommendations based on live data, formatted through phiData.
- News Agent: Summarizes current financial news impacting stock prices, using phiData to frame search queries and results.
- Portfolio Agent: Advises on portfolio optimization strategies, utilizing phiData for structuring portfolio data.
- Risk Management Agent: Analyzes and suggests mitigation strategies for investment risks, using phiData to maintain risk data consistency.
- Sentiment Analysis Agent: Evaluates market sentiment using news and social media data, structured via phiData for effective sentiment analysis.
- Multi-Agent Team
- Agent Team: Coordinates among individual agents to handle complex queries, ensuring seamless integration and comprehensive response generation with phiData aiding in data structuring and communication among agents.

**Usage**
The addition of phiData provides a sophisticated framework that ensures data integrity and enhances the capabilities of agents to interact and process complex datasets seamlessly. 
