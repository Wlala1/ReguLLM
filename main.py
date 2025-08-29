from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi

# 初始化 LLM 和 嵌入模型
llm = ChatTongyi(model="qwen-max", temperature=0.1)

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# 加载已经存在的向量数据库
vector_store = Chroma(persist_directory="./legal_compliance_db", embedding_function=embedding_model)

# 创建一个检索器 (Retriever)，它可以根据查询找到相关文档
retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # k=3 表示返回最相关的3个文档片段
feature_description = ''
def tst(feature_description):

    retrieved_docs = retriever.invoke(feature_description) 
    print(retrieved_docs)

    # 设计一个 Prompt 模板
    template = """
    You are an expert legal compliance analyst. Your task is to determine if a feature needs geo-specific compliance logic based on the provided context and feature description.
    You should first identify the location of the feature and find related acts. You must distinguish between legal requirements and business decisions. Output your answer in a valid JSON format.

    CONTEXT FROM KNOWLEDGE BASE:
    {context}

    FEATURE DESCRIPTION:
    {feature_description}

    Based on the context and the description, provide your analysis as a JSON object with the keys 'needs_compliance_logic' (boolean), 'reasoning' (string), and 'identified_regulation' (string, or null).
    """

    prompt = PromptTemplate.from_template(template)

    # 使用 LangChain 的 LCEL 链式语法将所有步骤串起来
    rag_chain = (
        {"context": retriever, "feature_description": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 运行 RAG 链
    result_json = rag_chain.invoke(feature_description)
    print(result_json)

    return result_json, retrieved_docs
tst(feature_description)





data =['Curfew login blocker with ASL and GH for Utah minors',
       'PF default toggle with NR enforcement for California teens',
       'Child abuse content scanner using T5 and CDS triggers',
       'Content visibility lock with NSP for EU DSA',
       'Jellybean-based parental notifications for Florida regulation',
       'Unified retention control via DRT & CDS',
       'NSP auto-flagging',
       'T5 tagging for sensitive reports',
       'Underage protection via Snowcap trigger',
       'Universal PF deactivation on guest mode',
       'Story resharing with content expiry',
       'Leaderboard system for weekly creators',
       'Mood-based PF enhancements',
       'New user rewards via NR profile suggestions',
       'Creator fund payout tracking in CDS',
       'Trial run of video replies in EU',
       'Canada-first PF variant test',
       'Chat UI overhaul',
       'Regional trial of autoplay behavior',
       'South Korea dark theme A/B experiment',
       'Age-specific notification controls with ASL',
       'Chat content restrictions via LCP',
       'Video upload limits for new users',
       'Flag escalation flow for sensitive comments',
       'User behavior scoring for policy gating',
       'Minor-safe chat expansion via Jellybean',
       'Friend suggestions with underage safeguards',
       'Reaction GIFs with embedded filtering',
       'Longform posts with age-based moderation',
       'Custom avatar system with identity checks']

description = [
    "To comply with the Utah Social Media Regulation Act, we are implementing a curfew-based login restriction for users under 18. The system uses ASL to detect minor accounts and routes enforcement through GH to apply only within Utah boundaries. The feature activates during restricted night hours and logs activity using EchoTrace for auditability. This allows parental control to be enacted without user-facing alerts, operating in ShadowMode during initial rollout.",
    "As part of compliance with California’s SB976, the app will disable PF by default for users under 18 located in California. This default setting is considered NR to override, unless explicit parental opt-in is provided. Geo-detection is handled via GH, and rollout is monitored with FR logs. The design ensures minimal disruption while meeting the strict personalization requirements imposed by the law.",
    "In line with the US federal law requiring providers to report child sexual abuse content to NCMEC, this feature scans uploads and flags suspected materials tagged as T5. Once flagged, the CDS auto-generates reports and routes them via secure channel APIs. The logic runs in real-time, supports human validation, and logs detection metadata for internal audits. Regional thresholds are governed by LCP parameters in the backend.",
    "To meet the transparency expectations of the EU Digital Services Act, we are introducing a visibility lock for flagged user-generated content labeled under NSP. When such content is detected, a soft Softblock is applied and GH ensures enforcement is restricted to the EU region only. EchoTrace supports traceability, and Redline status can be triggered for legal review. This feature enhances accountability and complies with Article 16’s removal mechanisms.",
    "To support Florida's Online Protections for Minors law, this feature extends the Jellybean parental control framework. Notifications are dispatched to verified parent accounts when a minor attempts to access restricted features. Using IMT, the system checks behavioral anomalies against BB models. If violations are detected, restrictions are applied in ShadowMode with full audit logging through CDS. Glow flags ensure compliance visibility during rollout phases.",
    "Introduce a data retention feature using DRT thresholds, ensuring automatic log deletion across all regions. CDS will continuously audit retention violations, triggering EchoTrace as necessary. Spanner logic ensures all platform modules comply uniformly.",
    "This feature will automatically detect and tag content that violates NSP policy. Once flagged, Softblock is applied and a Redline alert is generated if downstream sharing is attempted.",
    "When users report content containing high-risk information, it is tagged as T5 for internal routing. CDS then enforces escalation. The system is universal and does not rely on regional toggles or GH routes.",
    "Snowcap is activated for all underage users platform-wide, applying ASL to segment accounts. Actions taken under this logic are routed to CDS and monitored using BB to identify deviations in usage.",
    "By default, PF will be turned off for all uses browsing in guest mode.",
    "Enable users to reshare stories from others, with auto-expiry after 48 hours. This feature logs resharing attempts with EchoTrace and stores activity under BB.",
    "Introduce a creator leaderboard updated weekly using internal analytics. Points and rankings are stored in FR metadata and tracked using IMT.",
    "Adjust PF recommendations based on inferred mood signals from emoji usage. This logic is soft-tuned using BB and undergoes quiet testing in ShadowMode.",
    "At onboarding, users will receive NR-curated profiles to follow for faster network building. A/B testing will use Spanner.",
    "Monetization events will be tracked through CDS to detect anomalies in creator payouts. DRT rules apply for log trimming.",
    "Roll out video reply functionality to users in EEA only. GH will manage exposure control, and BB is used to baseline feedback.",
    "Launch a PF variant in CA as part of early experimentation. Spanner will isolate affected cohorts and Glow flags will monitor feature health.",
    "A new chat layout will be tested in the following regions: CA, US, BR, ID. GH will ensure location targeting and ShadowMode will collect usage metrics without user impact.",
    "Enable video autoplay only for users in US. GH filters users, while Spanner logs click-through deltas.",
    "A/B test dark theme accessibility for users in South Korea. Rollout is limited via GH and monitored with FR flags.",
    "Notifications will be tailored by age using ASL, allowing us to throttle or suppress push alerts for minors. EchoTrace will log adjustments, and CDS will verify enforcement across rollout waves.",
    "Enforce message content constraints by injecting LCP rules on delivery. ShadowMode will initially deploy the logic for safe validation. No explicit mention of legal requirements, but privacy context is implied.",
    "Introduce limits on video uploads from new accounts. IMT will trigger thresholds based on BB patterns. These limitations are partly for platform safety but without direct legal mapping.",
    "A flow that detects high-risk comment content and routes it via CDS with Redline markers. The logic applies generally and is monitored through EchoTrace, with no mention of regional policies.",
    "Behavioral scoring via Spanner will be used to gate access to certain tools. The feature tracks usage and adjusts gating based on BB divergence.",
    "We’re expanding chat features, but for users flagged by Jellybean, certain functions (e.g., media sharing) will be limited. BB and ASL will monitor compliance posture.",
    "New suggestion logic uses PF to recommend friends, but minors are excluded from adult pools using ASL and CDS logic. EchoTrace logs interactions in case future policy gates are needed.",
    "Enable GIFs in comments, while filtering content deemed inappropriate for minor accounts. Softblock will apply if a flagged GIF is used by ASL-flagged profiles.",
    "Longform post creation is now open to all. However, moderation for underage authors is stricter via Snowcap.",
    "Users can now design custom avatars. For safety, T5 triggers block adult-themed assets from use by underage profiles. Age detection uses ASL and logs flow through GH."
]
from langchain.schema import Document
import json

# class DocumentJSONEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, Document):
#             return {
#                 'page_content': obj.page_content,
#                 'metadata': obj.metadata
#             }
#         return super().default(obj)
# result = []
# for i in range(30):
#     inputs = data[i] + ": " + description[i]
#     tst_res, retrive = tst(inputs)
#     temp = {}
#     temp['input'] = inputs
#     temp['output1'] = tst_res
#     temp['retrive'] = retrive  # 直接保存，不需要预处理
#     result.append(temp)

# # 使用自定义编码器保存
# with open('result_tongyi.json', 'w', encoding='utf-8') as f:
#     json.dump(result, f, indent=4, ensure_ascii=False, cls=DocumentJSONEncoder)
