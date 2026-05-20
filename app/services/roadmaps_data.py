"""
Curated roadmap data extracted from roadmap.sh (MIT Licensed).
Structure: each roadmap has phases → topics → subtopics.
LLM adds time estimates + Indian market context on top of this.
"""

ROADMAPS = {
    # ── IT / Tech roles ────────────────────────────────────────────────────────

    "frontend": {
        "title": "Frontend Developer",
        "aliases": ["frontend developer", "front end developer", "frontend engineer", "front-end developer", "ui developer", "react developer", "vue developer", "angular developer"],
        "phases": [
            {
                "phase": "Foundations",
                "topics": [
                    {"name": "HTML", "subtopics": ["Semantic HTML", "Forms & Validations", "Accessibility", "SEO Basics", "Meta Tags"]},
                    {"name": "CSS", "subtopics": ["Selectors & Specificity", "Box Model", "Flexbox", "Grid", "Responsive Design", "Animations & Transitions"]},
                    {"name": "JavaScript Basics", "subtopics": ["Variables & Data Types", "DOM Manipulation", "Fetch API & AJAX", "ES6+ Features", "Event Handling", "Error Handling"]},
                ]
            },
            {
                "phase": "Version Control & Tools",
                "topics": [
                    {"name": "Git & GitHub", "subtopics": ["Git Basics", "Branching & Merging", "Pull Requests", "GitHub Actions Basics"]},
                    {"name": "Package Managers", "subtopics": ["npm", "yarn", "pnpm"]},
                    {"name": "Build Tools", "subtopics": ["Vite", "Webpack Basics", "Babel"]},
                ]
            },
            {
                "phase": "Modern Frameworks",
                "topics": [
                    {"name": "React.js", "subtopics": ["Components & Props", "State & Lifecycle", "Hooks (useState, useEffect)", "Context API", "React Router", "Performance Optimization"]},
                    {"name": "TypeScript", "subtopics": ["Types & Interfaces", "Generics", "Type Narrowing", "TypeScript with React"]},
                    {"name": "State Management", "subtopics": ["Redux Toolkit", "Zustand", "React Query"]},
                ]
            },
            {
                "phase": "Styling & UI",
                "topics": [
                    {"name": "CSS Frameworks", "subtopics": ["Tailwind CSS", "Styled Components", "CSS Modules"]},
                    {"name": "Component Libraries", "subtopics": ["Material UI", "Shadcn UI", "Ant Design"]},
                ]
            },
            {
                "phase": "Testing & Deployment",
                "topics": [
                    {"name": "Testing", "subtopics": ["Jest", "React Testing Library", "Cypress for E2E"]},
                    {"name": "Performance", "subtopics": ["Lighthouse", "Core Web Vitals", "Code Splitting", "Lazy Loading"]},
                    {"name": "Deployment", "subtopics": ["Vercel", "Netlify", "CI/CD Basics", "Docker Basics"]},
                ]
            },
        ]
    },

    "backend": {
        "title": "Backend Developer",
        "aliases": ["backend developer", "back end developer", "backend engineer", "server side developer", "node developer", "python backend", "java backend"],
        "phases": [
            {
                "phase": "Language & Basics",
                "topics": [
                    {"name": "Choose a Language", "subtopics": ["Python (Django/Flask/FastAPI)", "Node.js (Express)", "Java (Spring Boot)", "Go"]},
                    {"name": "OS & Networking", "subtopics": ["How the Internet Works", "HTTP/HTTPS", "DNS", "TCP/IP Basics", "SSH"]},
                ]
            },
            {
                "phase": "Databases",
                "topics": [
                    {"name": "Relational Databases", "subtopics": ["SQL Basics", "PostgreSQL", "Joins & Indexes", "Transactions & ACID", "Query Optimization"]},
                    {"name": "NoSQL Databases", "subtopics": ["MongoDB", "Redis (Caching)", "Database Scaling Basics"]},
                    {"name": "ORMs", "subtopics": ["SQLAlchemy (Python)", "Prisma (Node)", "Hibernate (Java)"]},
                ]
            },
            {
                "phase": "APIs",
                "topics": [
                    {"name": "REST APIs", "subtopics": ["REST Principles", "HTTP Methods", "Status Codes", "Authentication (JWT, OAuth2)", "API Versioning"]},
                    {"name": "GraphQL", "subtopics": ["Schema & Resolvers", "Queries & Mutations", "GraphQL vs REST"]},
                    {"name": "API Security", "subtopics": ["CORS", "Rate Limiting", "Input Validation", "OWASP Top 10"]},
                ]
            },
            {
                "phase": "Advanced Backend",
                "topics": [
                    {"name": "Caching", "subtopics": ["Redis", "CDN Caching", "Cache Invalidation Strategies"]},
                    {"name": "Message Queues", "subtopics": ["RabbitMQ", "Apache Kafka", "Celery (Python)"]},
                    {"name": "Microservices", "subtopics": ["Service Discovery", "API Gateway", "Event-Driven Architecture"]},
                ]
            },
            {
                "phase": "DevOps & Deployment",
                "topics": [
                    {"name": "Containers", "subtopics": ["Docker", "Docker Compose", "Container Registries"]},
                    {"name": "Cloud", "subtopics": ["AWS (EC2, S3, RDS, Lambda)", "GCP Basics", "Serverless"]},
                    {"name": "CI/CD", "subtopics": ["GitHub Actions", "Jenkins Basics", "Deployment Pipelines"]},
                ]
            },
        ]
    },

    "full-stack": {
        "title": "Full Stack Developer",
        "aliases": ["full stack developer", "fullstack developer", "full-stack developer", "full stack engineer", "mern developer", "mean developer"],
        "phases": [
            {"phase": "Frontend Basics", "topics": [
                {"name": "HTML, CSS & JavaScript", "subtopics": ["Semantic HTML", "CSS Flexbox & Grid", "ES6+ JS", "DOM Manipulation"]},
                {"name": "React.js", "subtopics": ["Components & Hooks", "State Management", "React Router", "Context API"]},
            ]},
            {"phase": "Backend Basics", "topics": [
                {"name": "Node.js & Express", "subtopics": ["REST API Design", "Middleware", "Authentication (JWT)", "Error Handling"]},
                {"name": "Databases", "subtopics": ["MongoDB (Mongoose)", "PostgreSQL", "SQL vs NoSQL Decision"]},
            ]},
            {"phase": "Full Stack Integration", "topics": [
                {"name": "API Integration", "subtopics": ["Axios / Fetch", "CORS Handling", "WebSockets", "File Uploads"]},
                {"name": "Authentication", "subtopics": ["JWT Auth Flow", "OAuth2 / Google Login", "Session Management", "Role-Based Access"]},
            ]},
            {"phase": "DevOps & Deployment", "topics": [
                {"name": "Docker & CI/CD", "subtopics": ["Dockerizing Apps", "GitHub Actions", "Environment Variables"]},
                {"name": "Cloud Deployment", "subtopics": ["AWS / GCP / Azure", "Vercel (Frontend)", "Railway / Render (Backend)"]},
            ]},
        ]
    },

    "devops": {
        "title": "DevOps Engineer",
        "aliases": ["devops engineer", "devops", "sre", "site reliability engineer", "platform engineer", "cloud engineer", "infrastructure engineer"],
        "phases": [
            {"phase": "OS & Networking", "topics": [
                {"name": "Linux", "subtopics": ["File System", "Shell Scripting (Bash)", "Process Management", "Permissions & Users", "Cron Jobs", "SSH"]},
                {"name": "Networking", "subtopics": ["TCP/IP", "DNS", "HTTP/HTTPS", "Load Balancers", "Firewalls & VPNs"]},
            ]},
            {"phase": "Version Control & CI/CD", "topics": [
                {"name": "Git & GitHub", "subtopics": ["Advanced Git", "GitOps", "Branching Strategies"]},
                {"name": "CI/CD Pipelines", "subtopics": ["GitHub Actions", "Jenkins", "GitLab CI", "CircleCI", "Deployment Strategies (Blue-Green, Canary)"]},
            ]},
            {"phase": "Containers & Orchestration", "topics": [
                {"name": "Docker", "subtopics": ["Dockerfile", "Docker Compose", "Container Networking", "Image Optimization"]},
                {"name": "Kubernetes", "subtopics": ["Pods & Deployments", "Services & Ingress", "ConfigMaps & Secrets", "Helm Charts", "HPA & Scaling"]},
            ]},
            {"phase": "Cloud Platforms", "topics": [
                {"name": "AWS", "subtopics": ["EC2, S3, RDS", "IAM & Security", "VPC & Networking", "Lambda", "CloudWatch"]},
                {"name": "Infrastructure as Code", "subtopics": ["Terraform", "Ansible", "Pulumi Basics"]},
            ]},
            {"phase": "Monitoring & Security", "topics": [
                {"name": "Monitoring", "subtopics": ["Prometheus", "Grafana", "ELK Stack", "PagerDuty Alerts"]},
                {"name": "Security", "subtopics": ["DevSecOps", "SAST/DAST", "Secrets Management (Vault)", "Compliance Basics"]},
            ]},
        ]
    },

    "data-analyst": {
        "title": "Data Analyst",
        "aliases": ["data analyst", "business analyst", "analytics engineer", "bi analyst", "data analytics", "business intelligence analyst"],
        "phases": [
            {"phase": "Data Foundations", "topics": [
                {"name": "Excel & Google Sheets", "subtopics": ["Pivot Tables", "VLOOKUP / XLOOKUP", "Data Cleaning", "Charts & Dashboards", "Macros Basics"]},
                {"name": "SQL", "subtopics": ["SELECT, WHERE, GROUP BY", "JOINs (Inner, Left, Right)", "Subqueries & CTEs", "Window Functions", "Query Optimization"]},
            ]},
            {"phase": "Programming", "topics": [
                {"name": "Python for Analysis", "subtopics": ["Pandas (DataFrames)", "NumPy", "Data Cleaning & EDA", "Matplotlib & Seaborn", "Jupyter Notebooks"]},
            ]},
            {"phase": "Visualization & BI", "topics": [
                {"name": "Power BI", "subtopics": ["Data Modeling", "DAX Formulas", "Interactive Dashboards", "Power Query (ETL)"]},
                {"name": "Tableau", "subtopics": ["Connecting Data Sources", "Calculated Fields", "Dashboards & Stories", "Tableau Public"]},
            ]},
            {"phase": "Statistics & Analytics", "topics": [
                {"name": "Statistics", "subtopics": ["Descriptive Statistics", "Probability Basics", "Hypothesis Testing", "A/B Testing", "Regression Analysis"]},
                {"name": "Advanced Analytics", "subtopics": ["Cohort Analysis", "Funnel Analysis", "RFM Analysis", "Google Analytics / Mixpanel"]},
            ]},
            {"phase": "Data Engineering Basics", "topics": [
                {"name": "Databases & ETL", "subtopics": ["PostgreSQL", "BigQuery", "Snowflake Basics", "ETL Pipelines", "dbt (Data Build Tool)"]},
            ]},
        ]
    },

    "ai-engineer": {
        "title": "AI Engineer",
        "aliases": ["ai engineer", "artificial intelligence engineer", "ai developer", "llm engineer", "genai engineer", "generative ai engineer", "ai ml engineer"],
        "phases": [
            {"phase": "Math & Programming", "topics": [
                {"name": "Math for AI", "subtopics": ["Linear Algebra (Vectors, Matrices)", "Calculus (Gradients, Backprop)", "Probability & Statistics", "Information Theory Basics"]},
                {"name": "Python for AI", "subtopics": ["NumPy & Pandas", "Matplotlib & Seaborn", "Jupyter Notebooks", "OOP in Python"]},
            ]},
            {"phase": "ML Foundations", "topics": [
                {"name": "Machine Learning", "subtopics": ["Supervised Learning", "Unsupervised Learning", "Model Evaluation & Metrics", "Cross-Validation", "Feature Engineering", "Scikit-learn"]},
                {"name": "Deep Learning", "subtopics": ["Neural Networks", "CNNs", "RNNs & LSTMs", "Transformers Architecture", "PyTorch / TensorFlow"]},
            ]},
            {"phase": "LLMs & Generative AI", "topics": [
                {"name": "LLMs & Prompt Engineering", "subtopics": ["How LLMs Work", "Prompt Engineering", "Few-shot & Zero-shot", "Chain of Thought", "OpenAI / Anthropic APIs"]},
                {"name": "RAG & Vector Databases", "subtopics": ["Embeddings", "Vector Databases (Pinecone, Weaviate)", "RAG Architecture", "LangChain / LlamaIndex"]},
                {"name": "Fine-Tuning", "subtopics": ["Transfer Learning", "LoRA / QLoRA", "RLHF Basics", "Hugging Face Transformers"]},
            ]},
            {"phase": "AI in Production", "topics": [
                {"name": "MLOps", "subtopics": ["Model Serving (FastAPI)", "MLflow / Weights & Biases", "Model Monitoring", "Docker for ML", "CI/CD for ML Pipelines"]},
                {"name": "AI Agents", "subtopics": ["Agentic Architecture", "Tool Use & Function Calling", "Multi-Agent Systems", "LangGraph / AutoGen"]},
            ]},
        ]
    },

    "machine-learning": {
        "title": "Machine Learning Engineer",
        "aliases": ["machine learning engineer", "ml engineer", "ml developer", "machine learning developer", "deep learning engineer", "nlp engineer", "computer vision engineer"],
        "phases": [
            {"phase": "Math Foundations", "topics": [
                {"name": "Mathematics", "subtopics": ["Linear Algebra (Vectors, Matrices, Eigenvalues)", "Calculus (Derivatives, Chain Rule, Backpropagation)", "Probability & Statistics", "Bayesian Thinking", "Information Theory"]},
            ]},
            {"phase": "Programming & Data", "topics": [
                {"name": "Python for ML", "subtopics": ["NumPy Arrays & Broadcasting", "Pandas DataFrames & Data Wrangling", "Matplotlib & Seaborn Visualization", "Scikit-learn Pipelines", "Jupyter Notebooks"]},
                {"name": "Data Engineering", "subtopics": ["Data Collection & Cleaning", "Feature Engineering", "Feature Scaling & Encoding", "Handling Imbalanced Data", "Train/Val/Test Splits"]},
            ]},
            {"phase": "Core ML Algorithms", "topics": [
                {"name": "Supervised Learning", "subtopics": ["Linear & Logistic Regression", "Decision Trees & Random Forests", "SVM", "Gradient Boosting (XGBoost, LightGBM)", "K-Nearest Neighbors"]},
                {"name": "Unsupervised Learning", "subtopics": ["K-Means Clustering", "DBSCAN", "PCA & Dimensionality Reduction", "Anomaly Detection"]},
                {"name": "Model Evaluation", "subtopics": ["Confusion Matrix, Precision, Recall, F1", "ROC-AUC Curve", "Cross-Validation", "Hyperparameter Tuning (GridSearch, Optuna)"]},
            ]},
            {"phase": "Deep Learning", "topics": [
                {"name": "Neural Networks", "subtopics": ["Feedforward Networks", "Activation Functions", "Loss Functions", "Optimizers (SGD, Adam)", "Regularization (Dropout, BatchNorm)"]},
                {"name": "CNNs", "subtopics": ["Convolution Operations", "Pooling Layers", "ResNet / VGG / EfficientNet", "Transfer Learning", "Object Detection (YOLO)"]},
                {"name": "NLP & Transformers", "subtopics": ["Word Embeddings (Word2Vec, GloVe)", "Seq2Seq Models", "Attention Mechanism", "BERT & GPT Architecture", "Hugging Face Transformers"]},
                {"name": "Frameworks", "subtopics": ["PyTorch (preferred)", "TensorFlow / Keras", "GPU Training Basics"]},
            ]},
            {"phase": "MLOps & Deployment", "topics": [
                {"name": "ML in Production", "subtopics": ["Model Serialization (Pickle, ONNX)", "FastAPI for Model Serving", "Docker for ML", "MLflow Experiment Tracking", "Model Monitoring & Drift Detection"]},
                {"name": "Cloud ML", "subtopics": ["AWS SageMaker", "GCP Vertex AI", "Databricks Basics", "Airflow for ML Pipelines"]},
            ]},
        ]
    },

    "data-engineer": {
        "title": "Data Engineer",
        "aliases": ["data engineer", "data pipeline engineer", "etl developer", "big data engineer", "analytics engineer"],
        "phases": [
            {"phase": "Programming", "topics": [
                {"name": "Python for Data", "subtopics": ["Pandas & NumPy", "File I/O (CSV, JSON, Parquet)", "Python OOP", "Virtual Environments"]},
                {"name": "SQL & Databases", "subtopics": ["Advanced SQL", "PostgreSQL", "Indexing & Query Optimization", "Database Design"]},
            ]},
            {"phase": "Data Pipeline Tools", "topics": [
                {"name": "ETL & Orchestration", "subtopics": ["Apache Airflow (DAGs, Tasks, Scheduling)", "dbt (Data Build Tool)", "Prefect / Dagster"]},
                {"name": "Big Data", "subtopics": ["Apache Spark (PySpark)", "Hadoop HDFS Basics", "Batch vs Streaming Processing"]},
                {"name": "Streaming", "subtopics": ["Apache Kafka", "Apache Flink Basics", "Real-time Pipelines"]},
            ]},
            {"phase": "Data Warehousing", "topics": [
                {"name": "Cloud Warehouses", "subtopics": ["Snowflake", "Google BigQuery", "AWS Redshift", "Data Modeling (Star/Snowflake Schema)"]},
                {"name": "Data Lake", "subtopics": ["AWS S3 / GCS", "Delta Lake", "Apache Iceberg", "Data Lakehouse Architecture"]},
            ]},
            {"phase": "Cloud & DevOps", "topics": [
                {"name": "Cloud Platforms", "subtopics": ["AWS (S3, Glue, EMR, Lambda)", "GCP (Dataflow, Dataproc, BigQuery)", "Azure (Data Factory, Synapse)"]},
                {"name": "Infrastructure", "subtopics": ["Docker for Data Pipelines", "Terraform", "CI/CD for Data Pipelines", "Monitoring (Great Expectations, Monte Carlo)"]},
            ]},
        ]
    },

    "android": {
        "title": "Android Developer",
        "aliases": ["android developer", "android engineer", "kotlin developer", "android app developer", "mobile developer android"],
        "phases": [
            {"phase": "Kotlin & Basics", "topics": [
                {"name": "Kotlin", "subtopics": ["Syntax & Variables", "OOP in Kotlin", "Coroutines & Flow", "Null Safety", "Extension Functions"]},
                {"name": "Android Fundamentals", "subtopics": ["Activity & Fragment Lifecycle", "Intents", "Permissions", "RecyclerView", "View Binding"]},
            ]},
            {"phase": "Modern Android", "topics": [
                {"name": "Jetpack Compose", "subtopics": ["Composable Functions", "State in Compose", "Navigation", "LazyColumn", "Theming"]},
                {"name": "Architecture", "subtopics": ["MVVM Pattern", "LiveData & ViewModel", "Repository Pattern", "Hilt (Dependency Injection)"]},
            ]},
            {"phase": "Data & Networking", "topics": [
                {"name": "Networking", "subtopics": ["Retrofit & OkHttp", "REST API Integration", "Coroutines for Async", "Gson / Moshi Parsing"]},
                {"name": "Local Storage", "subtopics": ["Room Database", "SharedPreferences / DataStore", "File Storage"]},
            ]},
            {"phase": "Publishing", "topics": [
                {"name": "Deployment", "subtopics": ["Google Play Console", "Signing APK/AAB", "App Bundle", "ProGuard & R8", "Firebase (Crashlytics, Analytics)"]},
            ]},
        ]
    },

    "ios": {
        "title": "iOS Developer",
        "aliases": ["ios developer", "ios engineer", "swift developer", "iphone developer", "apple developer", "mobile developer ios"],
        "phases": [
            {"phase": "Swift & Foundations", "topics": [
                {"name": "Swift", "subtopics": ["Syntax & Variables", "Optionals", "Closures", "Protocols & Extensions", "Generics", "Concurrency (async/await)"]},
                {"name": "UIKit", "subtopics": ["View Controllers", "Auto Layout", "Table & Collection Views", "Navigation Controller", "Storyboards vs Programmatic UI"]},
            ]},
            {"phase": "SwiftUI", "topics": [
                {"name": "SwiftUI", "subtopics": ["Views & Modifiers", "State & Binding", "Navigation", "Lists & ForEach", "Animations", "MVVM with SwiftUI"]},
            ]},
            {"phase": "Data & Networking", "topics": [
                {"name": "Networking", "subtopics": ["URLSession", "Codable for JSON", "Async/Await Networking", "Third-party libs (Alamofire)"]},
                {"name": "Persistence", "subtopics": ["UserDefaults", "CoreData", "SwiftData", "Keychain"]},
            ]},
            {"phase": "Publishing", "topics": [
                {"name": "App Store", "subtopics": ["Xcode Signing & Certificates", "App Store Connect", "TestFlight", "App Review Guidelines", "In-App Purchases"]},
            ]},
        ]
    },

    "cyber-security": {
        "title": "Cyber Security Engineer",
        "aliases": ["cyber security", "cybersecurity engineer", "security engineer", "information security", "ethical hacker", "penetration tester", "security analyst"],
        "phases": [
            {"phase": "Foundations", "topics": [
                {"name": "Networking", "subtopics": ["OSI Model", "TCP/IP", "DNS, HTTP, FTP, SSH", "Firewalls & IDS/IPS", "Wireshark"]},
                {"name": "OS & Linux", "subtopics": ["Linux Administration", "File Permissions", "Process Management", "Bash Scripting", "Windows Security Basics"]},
            ]},
            {"phase": "Security Concepts", "topics": [
                {"name": "Cryptography", "subtopics": ["Symmetric & Asymmetric Encryption", "Hashing (SHA, MD5)", "TLS/SSL", "PKI & Certificates", "Zero Knowledge Proofs"]},
                {"name": "OWASP & Web Security", "subtopics": ["SQL Injection", "XSS", "CSRF", "Authentication Flaws", "Security Headers", "Burp Suite"]},
            ]},
            {"phase": "Offensive Security", "topics": [
                {"name": "Penetration Testing", "subtopics": ["Reconnaissance (OSINT)", "Scanning (Nmap, Nessus)", "Exploitation (Metasploit)", "Post Exploitation", "Reporting"]},
                {"name": "CTF & Practice", "subtopics": ["TryHackMe / HackTheBox", "Reverse Engineering Basics", "Buffer Overflow", "Privilege Escalation"]},
            ]},
            {"phase": "Defensive Security", "topics": [
                {"name": "SOC & Incident Response", "subtopics": ["SIEM (Splunk, Elastic)", "Log Analysis", "Threat Intelligence", "Incident Response Playbooks", "Digital Forensics Basics"]},
                {"name": "Cloud Security", "subtopics": ["AWS IAM Security", "Zero Trust Architecture", "CSPM Tools", "Compliance (ISO 27001, SOC2)"]},
            ]},
        ]
    },

    "blockchain": {
        "title": "Blockchain Developer",
        "aliases": ["blockchain developer", "web3 developer", "smart contract developer", "solidity developer", "crypto developer", "defi developer"],
        "phases": [
            {"phase": "Blockchain Basics", "topics": [
                {"name": "Blockchain Fundamentals", "subtopics": ["How Blockchain Works", "Consensus Mechanisms (PoW, PoS)", "Public vs Private Blockchain", "Cryptographic Hashing", "Merkle Trees"]},
                {"name": "Ethereum", "subtopics": ["Ethereum Architecture", "EVM", "Gas & Transactions", "Wallets (MetaMask)", "Etherscan"]},
            ]},
            {"phase": "Smart Contracts", "topics": [
                {"name": "Solidity", "subtopics": ["Data Types & Variables", "Functions & Modifiers", "Events & Logs", "Inheritance", "Error Handling", "Gas Optimization"]},
                {"name": "Development Tools", "subtopics": ["Hardhat", "Foundry", "OpenZeppelin Contracts", "IPFS", "Alchemy / Infura"]},
            ]},
            {"phase": "DeFi & NFTs", "topics": [
                {"name": "DeFi Protocols", "subtopics": ["ERC-20 Tokens", "Uniswap & AMMs", "Lending Protocols (Aave)", "Yield Farming", "Flash Loans"]},
                {"name": "NFTs & Web3", "subtopics": ["ERC-721 & ERC-1155", "NFT Marketplaces", "Web3.js / Ethers.js", "React + Web3 DApps", "IPFS for NFT Metadata"]},
            ]},
        ]
    },

    "qa": {
        "title": "QA Engineer",
        "aliases": ["qa engineer", "quality assurance", "test engineer", "sdet", "automation engineer", "software tester", "qa analyst"],
        "phases": [
            {"phase": "Testing Fundamentals", "topics": [
                {"name": "Manual Testing", "subtopics": ["SDLC & STLC", "Test Cases & Test Plans", "Types of Testing (Functional, Regression, Smoke)", "Bug Reporting", "Agile Testing"]},
            ]},
            {"phase": "Automation", "topics": [
                {"name": "Selenium", "subtopics": ["WebDriver Setup", "Locators (XPath, CSS)", "Page Object Model", "Test Suites", "Selenium Grid"]},
                {"name": "API Testing", "subtopics": ["Postman", "REST Assured", "API Test Automation", "Contract Testing"]},
                {"name": "Modern Frameworks", "subtopics": ["Cypress", "Playwright", "Pytest", "TestNG / JUnit"]},
            ]},
            {"phase": "Performance & Security", "topics": [
                {"name": "Performance Testing", "subtopics": ["JMeter", "Load vs Stress Testing", "Response Time Analysis", "k6"]},
                {"name": "Mobile Testing", "subtopics": ["Appium", "Android Studio Emulator", "TestFlight", "BrowserStack"]},
            ]},
            {"phase": "CI/CD & DevOps QA", "topics": [
                {"name": "DevOps for QA", "subtopics": ["Jenkins Integration", "GitHub Actions for Tests", "Docker for Test Environments", "Test Reporting (Allure)"]},
            ]},
        ]
    },

    "product-manager": {
        "title": "Product Manager",
        "aliases": ["product manager", "pm", "product owner", "product lead", "associate product manager", "apm", "technical product manager", "tpm"],
        "phases": [
            {"phase": "PM Fundamentals", "topics": [
                {"name": "Product Thinking", "subtopics": ["Product vs Project Management", "User Problem Discovery", "Jobs To Be Done (JTBD)", "Product-Market Fit", "North Star Metric"]},
                {"name": "User Research", "subtopics": ["User Interviews", "Surveys & Questionnaires", "Usability Testing", "Persona Creation", "Empathy Mapping"]},
            ]},
            {"phase": "Strategy & Roadmapping", "topics": [
                {"name": "Product Strategy", "subtopics": ["Vision & Mission", "OKRs", "SWOT Analysis", "Competitive Analysis", "Porter's Five Forces"]},
                {"name": "Prioritization", "subtopics": ["RICE Framework", "MoSCoW Method", "Kano Model", "Opportunity Scoring", "Impact vs Effort Matrix"]},
            ]},
            {"phase": "Execution", "topics": [
                {"name": "Agile & Scrum", "subtopics": ["Sprints & Ceremonies", "User Stories & Acceptance Criteria", "Backlog Grooming", "Velocity & Burndown", "PRD Writing"]},
                {"name": "Stakeholder Management", "subtopics": ["Communicating with Engineering", "Executive Presentations", "Conflict Resolution", "Alignment Techniques"]},
            ]},
            {"phase": "Analytics & Growth", "topics": [
                {"name": "Data & Analytics", "subtopics": ["Product Metrics (DAU, MAU, Retention)", "Funnel Analysis", "A/B Testing", "SQL Basics for PMs", "Amplitude / Mixpanel"]},
                {"name": "Growth", "subtopics": ["Growth Hacking Basics", "Acquisition vs Retention", "Viral Loops", "Product-Led Growth (PLG)", "Pricing Strategies"]},
            ]},
        ]
    },

    "ux-design": {
        "title": "UX Designer",
        "aliases": ["ux designer", "ui ux designer", "ui/ux designer", "product designer", "interaction designer", "user experience designer", "visual designer"],
        "phases": [
            {"phase": "Design Foundations", "topics": [
                {"name": "Design Principles", "subtopics": ["Gestalt Principles", "Visual Hierarchy", "Color Theory", "Typography", "Spacing & Layout", "Accessibility (WCAG)"]},
                {"name": "UX Process", "subtopics": ["Design Thinking", "Double Diamond", "User Research Methods", "Affinity Mapping", "How Might We (HMW)"]},
            ]},
            {"phase": "Tools", "topics": [
                {"name": "Figma", "subtopics": ["Frames & Components", "Auto Layout", "Prototyping", "Design Systems", "Variables & Tokens", "Dev Mode"]},
            ]},
            {"phase": "UX Research & Testing", "topics": [
                {"name": "Research", "subtopics": ["User Interviews", "Usability Testing", "Card Sorting", "Tree Testing", "Surveys", "Heuristic Evaluation"]},
                {"name": "Information Architecture", "subtopics": ["Sitemaps", "User Flows", "Wireframing", "Low vs High Fidelity Prototypes"]},
            ]},
            {"phase": "UI & Portfolio", "topics": [
                {"name": "UI Design", "subtopics": ["Component Libraries", "Micro-interactions", "Motion Design Basics", "Responsive Design", "Dark Mode Design"]},
                {"name": "Portfolio & Career", "subtopics": ["Case Study Structure", "Behance / Dribbble", "Design Interviews", "Working with Developers"]},
            ]},
        ]
    },

    "software-architect": {
        "title": "Software Architect",
        "aliases": ["software architect", "solutions architect", "enterprise architect", "system architect", "technical architect", "cloud architect"],
        "phases": [
            {"phase": "Foundations", "topics": [
                {"name": "Design Patterns", "subtopics": ["SOLID Principles", "Creational Patterns (Factory, Singleton)", "Structural Patterns (Adapter, Facade)", "Behavioral Patterns (Observer, Strategy)", "Anti-patterns"]},
                {"name": "Clean Code", "subtopics": ["Code Readability", "Refactoring Techniques", "TDD & BDD", "Code Reviews", "Technical Debt Management"]},
            ]},
            {"phase": "System Design", "topics": [
                {"name": "Distributed Systems", "subtopics": ["CAP Theorem", "Consistency vs Availability", "Distributed Transactions", "Event Sourcing", "CQRS"]},
                {"name": "Scalability", "subtopics": ["Horizontal vs Vertical Scaling", "Load Balancing", "Database Sharding", "Read Replicas", "CDN"]},
                {"name": "Architecture Patterns", "subtopics": ["Monolith vs Microservices", "Serverless Architecture", "Event-Driven Architecture", "Hexagonal Architecture", "Service Mesh"]},
            ]},
            {"phase": "Cloud & Infrastructure", "topics": [
                {"name": "Cloud Architecture", "subtopics": ["AWS Well-Architected Framework", "Multi-Region Design", "Disaster Recovery", "Cost Optimization", "FinOps"]},
            ]},
        ]
    },

    "mlops": {
        "title": "MLOps Engineer",
        "aliases": ["mlops engineer", "ml platform engineer", "ai infrastructure engineer", "ml infrastructure"],
        "phases": [
            {"phase": "ML Foundations", "topics": [
                {"name": "Machine Learning Basics", "subtopics": ["Supervised & Unsupervised Learning", "Model Training & Evaluation", "Scikit-learn", "Feature Engineering"]},
                {"name": "Python & Dev Tools", "subtopics": ["Python OOP", "Virtual Environments", "Git for ML", "Jupyter Notebooks"]},
            ]},
            {"phase": "MLOps Core", "topics": [
                {"name": "Experiment Tracking", "subtopics": ["MLflow", "Weights & Biases", "DVC (Data Version Control)", "Experiment Comparison"]},
                {"name": "Model Serving", "subtopics": ["FastAPI for ML", "BentoML", "TorchServe", "Triton Inference Server", "Model Versioning"]},
                {"name": "Feature Stores", "subtopics": ["Feast", "Tecton", "Online vs Offline Features", "Feature Pipelines"]},
            ]},
            {"phase": "Infrastructure", "topics": [
                {"name": "Containers & Orchestration", "subtopics": ["Docker for ML", "Kubernetes for ML", "Kubeflow", "Argo Workflows"]},
                {"name": "Cloud ML Platforms", "subtopics": ["AWS SageMaker", "GCP Vertex AI", "Azure ML", "Databricks"]},
            ]},
            {"phase": "Monitoring & Governance", "topics": [
                {"name": "Model Monitoring", "subtopics": ["Data Drift Detection", "Model Performance Monitoring", "Grafana Dashboards", "Evidently AI", "Alerting"]},
                {"name": "ML Governance", "subtopics": ["Model Cards", "Responsible AI", "Bias Detection", "Model Lineage & Audit Trails"]},
            ]},
        ]
    },

    "network-engineer": {
        "title": "Network Engineer",
        "aliases": ["network engineer", "network administrator", "network analyst", "cisco engineer", "ccna", "ccnp", "network architect"],
        "phases": [
            {"phase": "Networking Fundamentals", "topics": [
                {"name": "OSI & TCP/IP Model", "subtopics": ["7 Layers of OSI", "TCP vs UDP", "IP Addressing", "Subnetting (CIDR)", "IPv4 vs IPv6"]},
                {"name": "Routing & Switching", "subtopics": ["Static vs Dynamic Routing", "OSPF", "BGP", "VLANs", "STP", "Trunking"]},
            ]},
            {"phase": "Network Devices", "topics": [
                {"name": "Cisco & Network Devices", "subtopics": ["Router Configuration (IOS)", "Switch Configuration", "Firewall Setup (ASA)", "ACLs", "NAT/PAT"]},
                {"name": "Wireless", "subtopics": ["WiFi Standards (802.11)", "WPA2/WPA3", "Wireless Controllers", "RF Interference"]},
            ]},
            {"phase": "Advanced Networking", "topics": [
                {"name": "Security", "subtopics": ["VPN (IPSec, SSL)", "Network Firewalls", "IDS/IPS", "Zero Trust Networking", "DDoS Mitigation"]},
                {"name": "Cloud Networking", "subtopics": ["AWS VPC", "Azure VNET", "SD-WAN", "Cloud Load Balancers", "Private Connectivity (Direct Connect)"]},
            ]},
        ]
    },

    "devsecops": {
        "title": "DevSecOps Engineer",
        "aliases": ["devsecops", "devsecops engineer", "security devops", "application security engineer", "appsec"],
        "phases": [
            {"phase": "DevOps + Security Basics", "topics": [
                {"name": "DevOps Fundamentals", "subtopics": ["CI/CD Pipelines", "Docker & Kubernetes", "Infrastructure as Code (Terraform)", "GitHub Actions"]},
                {"name": "Security Basics", "subtopics": ["OWASP Top 10", "Threat Modeling (STRIDE)", "Security Requirements", "Secure SDLC"]},
            ]},
            {"phase": "Security Testing", "topics": [
                {"name": "SAST & DAST", "subtopics": ["SAST (SonarQube, Semgrep)", "DAST (OWASP ZAP, Burp Suite)", "SCA (Snyk, Dependabot)", "Container Scanning (Trivy)", "Secrets Scanning (GitLeaks)"]},
                {"name": "Penetration Testing", "subtopics": ["Web App Pentesting", "API Security Testing", "Cloud Pentesting", "Red Team vs Blue Team"]},
            ]},
            {"phase": "Compliance & Governance", "topics": [
                {"name": "Compliance", "subtopics": ["ISO 27001", "SOC 2", "GDPR", "PCI-DSS", "Policy as Code (OPA)"]},
                {"name": "Secrets & Identity", "subtopics": ["HashiCorp Vault", "AWS IAM Best Practices", "RBAC", "Zero Trust", "Certificate Management"]},
            ]},
        ]
    },
}

# ── Role matcher ────────────────────────────────────────────────────────────────
def find_roadmap(target_role: str) -> dict | None:
    """
    Returns the matching roadmap dict if target_role matches any known role,
    else returns None (caller should use LLM fallback).
    """
    query = target_role.lower().strip()
    for key, roadmap in ROADMAPS.items():
        # Check title match
        if query in roadmap["title"].lower():
            return {"key": key, **roadmap}
        # Check aliases
        for alias in roadmap.get("aliases", []):
            if alias in query or query in alias:
                return {"key": key, **roadmap}
    return None


def get_all_role_titles() -> list[str]:
    """Returns list of all supported role titles for frontend autocomplete."""
    return [r["title"] for r in ROADMAPS.values()]