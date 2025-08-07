graph TB
    %% User Layer
    subgraph Users ["👥 User Layer"]
        Students["Data Science Students"]
        Professionals["Data Science Professionals"]
        SelfLearners["Self-Learners"]
    end

    %% Presentation Layer
    subgraph PresentationLayer ["🖥️ Presentation Layer"]
        WebApp["Web Application<br/>(Streamlit/Flask)"]
        CLI["CLI Interface"]
        Dashboard["Interactive Dashboard<br/>(Chart.js)"]
        FileUpload["File Upload System<br/>(2GB CSV/JSON)"]
    end

    %% API Gateway Layer
    subgraph APILayer ["🌐 API Gateway Layer"]
        FastAPI["FastAPI Server"]
        AuthMiddleware["Authentication<br/>Middleware"]
        RateLimiter["Rate Limiter"]
        RequestValidator["Request Validator"]
    end

    %% Orchestration Layer
    subgraph OrchestrationLayer ["🎯 Orchestration Layer"]
        CoordinatorAgent["Coordinator Agent<br/>(Task Router)"]
        LangGraphOrch["LangGraph Orchestrator<br/>(Agent Workflows)"]
        ContextManager["Context Manager<br/>(User History)"]
        TaskClassifier["Task Classifier<br/>(scikit-learn)"]
    end

    %% Multi-Agent System Core
    subgraph AgentSystem ["🤖 Multi-Agent System Core"]
        QAAgent["Q&A Agent<br/>• Theoretical Questions<br/>• Level-appropriate Explanations<br/>• AI/ML/DL Concepts"]
        
        PracticeAgent["Practice Problem Agent<br/>• Quiz Generation<br/>• Coding Challenges<br/>• Dataset-specific Tasks"]
        
        SocraticAgent["Socratic Dialogue Agent<br/>• Critical Thinking Questions<br/>• Guided Learning<br/>• Dataset Discussions"]
        
        DatasetAgent["Dataset Analysis Agent<br/>• Statistical Analysis<br/>• Preprocessing Suggestions<br/>• Algorithm Recommendations"]
        
        FeedbackAgent["Feedback Agent<br/>• Knowledge Base Updates<br/>• User Rating Processing<br/>• Trending Content Integration"]
    end

    %% RAG System
    subgraph RAGSystem ["🔍 RAG System"]
        VectorDB["FAISS Vector Database<br/>• Agent-specific Indexes<br/>• Similarity Search"]
        
        EmbeddingModel["Sentence-BERT<br/>• Text Embeddings<br/>• CPU Optimized"]
        
        LLMModels["Language Models<br/>• DistilBERT<br/>• TinyLLaMA<br/>• LoRA Fine-tuning"]
        
        QueryProcessor["Query Processor<br/>• Embedding Generation<br/>• Context Retrieval"]
    end

    %% Knowledge Base
    subgraph KnowledgeBase ["📚 Knowledge Base"]
        TextbooksDB["Textbooks<br/>(Deep Learning, ML)"]
        StackOverflowDB["Stack Overflow<br/>Q&A Database"]
        KaggleDB["Kaggle Notebooks<br/>& Datasets"]
        DocsDB["Documentation<br/>(scikit-learn, TensorFlow)"]
        ArXivDB["arXiv Papers<br/>Research Content"]
    end

    %% Processing Tools
    subgraph ProcessingTools ["🛠️ Processing Tools"]
        DataProcessing["Data Processing<br/>• Pandas (chunking for 2GB)<br/>• NumPy<br/>• Data Validation"]
        
        MLTools["ML Tools<br/>• scikit-learn<br/>• StandardScaler, KNNImputer<br/>• RandomForestClassifier"]
        
        Visualization["Visualization<br/>• Matplotlib/Seaborn<br/>• Histograms, Heatmaps<br/>• Correlation Plots"]
        
        CodeExecution["Code Execution<br/>• Docker Environment<br/>• Pylint Validation<br/>• Secure Sandboxing"]
        
        ImbalancedTools["Imbalanced Data<br/>• SMOTE<br/>• Over/Under Sampling<br/>• Class Weights"]
    end

    %% Data Storage
    subgraph DataStorage ["💾 Data Storage Layer"]
        SQLiteDB["SQLite Database<br/>• User Data<br/>• Feedback Storage<br/>• Session Management"]
        
        FileStorage["File Storage<br/>• User Datasets<br/>• Generated Reports<br/>• Cached Results"]
        
        CacheLayer["Cache Layer<br/>• Flask-Caching<br/>• Embedding Cache<br/>• Query Results"]
    end

    %% External Services
    subgraph ExternalServices ["🌍 External Services"]
        TwitterAPI["Twitter API<br/>(Trending Content)"]
        HuggingFace["Hugging Face<br/>(Model Hub)"]
        GitHubAPI["GitHub API<br/>(Code Examples)"]
    end

    %% Monitoring & Logging
    subgraph Monitoring ["📊 Monitoring & Logging"]
        MetricsCollector["Metrics Collector<br/>• Response Time<br/>• Accuracy Tracking<br/>• User Satisfaction"]
        
        Logger["Logging System<br/>• Error Tracking<br/>• Performance Logs<br/>• User Activity"]
        
        AlertSystem["Alert System<br/>• Performance Issues<br/>• Error Notifications<br/>• System Health"]
    end

    %% Deployment
    subgraph Deployment ["🚀 Deployment"]
        ContainerOrch["Container Orchestration<br/>(Docker)"]
        CloudPlatform["Cloud Platform<br/>(Heroku/Render)"]
        LocalDeployment["Local Deployment<br/>(PyInstaller)"]
    end

    %% User Interactions
    Users --> PresentationLayer
    
    %% Presentation to API
    PresentationLayer --> APILayer
    
    %% API to Orchestration
    APILayer --> OrchestrationLayer
    
    %% Orchestration to Agents
    OrchestrationLayer --> AgentSystem
    
    %% Agents to RAG
    AgentSystem --> RAGSystem
    
    %% RAG to Knowledge Base
    RAGSystem --> KnowledgeBase
    
    %% Agents to Tools
    AgentSystem --> ProcessingTools
    
    %% Data Flow
    AgentSystem --> DataStorage
    ProcessingTools --> DataStorage
    
    %% External Integrations
    FeedbackAgent --> ExternalServices
    
    %% Monitoring Connections
    APILayer --> Monitoring
    AgentSystem --> Monitoring
    ProcessingTools --> Monitoring
    
    %% Deployment Connections
    APILayer --> Deployment
    DataStorage --> Deployment
    
    %% Specific Agent Connections
    CoordinatorAgent --> QAAgent
    CoordinatorAgent --> PracticeAgent
    CoordinatorAgent --> SocraticAgent
    CoordinatorAgent --> DatasetAgent
    CoordinatorAgent --> FeedbackAgent
    
    %% RAG Connections
    QAAgent --> VectorDB
    PracticeAgent --> VectorDB
    SocraticAgent --> VectorDB
    DatasetAgent --> VectorDB
    FeedbackAgent --> VectorDB
    
    %% Data Processing Flow
    DatasetAgent --> DataProcessing
    DatasetAgent --> MLTools
    DatasetAgent --> Visualization
    DatasetAgent --> ImbalancedTools
    
    %% Feedback Loop
    FeedbackAgent --> VectorDB
    FeedbackAgent --> KnowledgeBase
    
    %% Context Management
    ContextManager --> SQLiteDB
    TaskClassifier --> CoordinatorAgent
    
    %% Styling
    classDef userLayer fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef presentationLayer fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef apiLayer fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef orchestrationLayer fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef agentLayer fill:#ffebee,stroke:#b71c1c,stroke-width:2px
    classDef ragLayer fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    classDef toolsLayer fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef dataLayer fill:#e3f2fd,stroke:#0d47a1,stroke-width:2px
    classDef externalLayer fill:#f9fbe7,stroke:#827717,stroke-width:2px
    classDef monitoringLayer fill:#fff8e1,stroke:#f57c00,stroke-width:2px
    classDef deploymentLayer fill:#fafafa,stroke:#424242,stroke-width:2px
    
    class Users,Students,Professionals,SelfLearners userLayer
    class PresentationLayer,WebApp,CLI,Dashboard,FileUpload presentationLayer
    class APILayer,FastAPI,AuthMiddleware,RateLimiter,RequestValidator apiLayer
    class OrchestrationLayer,CoordinatorAgent,LangGraphOrch,ContextManager,TaskClassifier orchestrationLayer
    class AgentSystem,QAAgent,PracticeAgent,SocraticAgent,DatasetAgent,FeedbackAgent agentLayer
    class RAGSystem,VectorDB,EmbeddingModel,LLMModels,QueryProcessor ragLayer
    class KnowledgeBase,TextbooksDB,StackOverflowDB,KaggleDB,DocsDB,ArXivDB ragLayer
    class ProcessingTools,DataProcessing,MLTools,Visualization,CodeExecution,ImbalancedTools toolsLayer
    class DataStorage,SQLiteDB,FileStorage,CacheLayer dataLayer
    class ExternalServices,TwitterAPI,HuggingFace,GitHubAPI externalLayer
    class Monitoring,MetricsCollector,Logger,AlertSystem monitoringLayer
    class Deployment,ContainerOrch,CloudPlatform,LocalDeployment deploymentLayer