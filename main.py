import os

from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(
    streaming=False,
    verbose=False,
    openai_api_key="EMPTY",
    openai_api_base="http://192.168.102.196:9998/v1/",
    model_name='Qwen2-7B-Instruct',
    temperature=0.7,
    max_tokens=28000,
)
print(llm.invoke('你好'))
from dotenv import load_dotenv
# 加载环境变量
load_dotenv()


from server.knowledge_base.kb_service.base import EmbeddingsFunAdapter, normalize

import os
from langchain.vectorstores.pgvector import PGVector
CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
    host=os.environ.get("PGVECTOR_HOST", "168.4.4.61"),
    port=int(os.environ.get("PGVECTOR_PORT", "5432")),
    database=os.environ.get("PGVECTOR_DATABASE", "langchain_sd"),
    user=os.environ.get("PGVECTOR_USER", "sd_test"),
    password=os.environ.get("PGVECTOR_PASSWORD", "sd_test%40ai"),
)

from langchain.embeddings import HuggingFaceBgeEmbeddings

embed_func = EmbeddingsFunAdapter("bge-large-zh")

db = PGVector(embedding_function=embed_func, connection_string=CONNECTION_STRING,collection_name="pzm_test",distance_strategy="cosine")


from langchain_core.output_parsers.json import parse_json_markdown
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate
import json
def get_custom_question(model: llm,
        query: str,
) -> dict:
    response_schemas = [
        ResponseSchema(name="year", description="If the question contains \"latest\", there is no need to return the year in the user's question."
                                                " If the year in the problem cannot be extracted, please return an empty string"),
        ResponseSchema(name="table_info", description="The table name of the database in the user question, if it is JSON, only returns the value of the name"),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(
        template="You are a little expert in analyzing user problems, specializing in breaking down user problems. "
                 "No need to answer user's questions, just break down the user's questions,and only the year and table name information that you want to return in the mode can be returned, please answer in Chinese."
                 "\n{format_instructions}\n{question}",
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions},
    )
    _input = prompt.format_prompt(question=query)
    # print(_input)
    output = model(_input.to_messages())
    json_obj = None
    try:
        # json_str_for_markdown = f"```json\n{output.content}\n```"
        json_obj = parse_json_markdown(output.content)
    except json.JSONDecodeError as e:
        json_obj = None
    year_info = ""
    table_info = ""
    query_type = ""
    if json_obj is not None:
        year_info = json_obj['year'] if 'year' in json_obj else ""
        table_info = json_obj['table_info'] if "table_info" in json_obj else ""
    # year_info = output['year'] if 'year' in json_obj else ""
    # table_info = output['table_info'] if "table_info" in json_obj else ""
    if table_info != '' and table_info[-1] != '表':
        table_info += '表'
    num_list = [i for i in year_info if str.isdigit(i)]
    year = ''.join(num_list)
    # return CustomQuestion(str(year), str(table_info))
    return str(year), str(table_info)

from langchain.vectorstores.pgvector import PGVector
# 参数类型，->str 返回数据类型
def get_real_table_name(year: int, table_info: str, embed_func:None, pg_vector:PGVector) -> str:
    sql_filter_table = {"type": {"eq": 1}}
    # if year != '':
    #     sql_filter_table["year"] = {"eq": year}
    # desc_text = "(langchain_pg_embedding.cmetadata->>'year')"
    desc_text = None
    docs = []
    if table_info !='':
        embeddings = embed_func.embed_query(table_info)
        docs = pg_vector.similarity_search_with_score_by_vector(embeddings, 10, sql_filter_table)
    # print(docs)for
    table_name: str = ''
    if len(docs) > 0:
        if year != '':
            for doc in docs:
                db_table_name = doc[0].metadata['table_name']
                db_year = doc[0].metadata['year']
                # 这里返回的db_year和db_table_name是字符串类型
                if year == db_year:
                    table_name = db_table_name
                    break
        else:
            heigh_year = 0
            first_distance = docs[0][1]
            db_table_name = docs[0][0].metadata['table_name']
            heigh_year = docs[0][0].metadata['year']
            docs = docs[1:]
            for doc in docs:
                distance = doc[1]
                difference = distance - first_distance
                if difference > 0.01:
                    continue
                db_table_name = doc[0].metadata['table_name']
                db_year = doc[0].metadata['year']
                if db_year >= heigh_year:
                    table_name = db_table_name
                heigh_year = db_year
            table_name = db_table_name
    return table_name

def get_field_info(table_name: str, pg_vector: PGVector, embed_func: None) -> list[dict]:
    field_info = []
    if table_name != '':
        sql_filter_field = {}
        sql_filter_field["type"] = {"eq": 2}
        sql_filter_field["table_name"] = {"eq": table_name}
        embeddings = embed_func.embed_query(table_name)
        docs = pg_vector.similarity_search_with_score_by_vector(embeddings, 100, sql_filter_field)
        # docs = pg_vector(embedding=None, k=100, sql_filter_table=sql_filter_field)

        for doc in docs:
            db_field_name = doc[0].metadata['field_name']
            db_field_desc = doc[0].page_content
            field_info.append({"field_name": db_field_name, "field_desc": db_field_desc})
            # field_info.append({"field_name": doc.metadata['field_name'],"field_desc":doc.page_content})
    return field_info


# %%
def get_type_by_question(
        model: llm,
        query: str
) -> dict:
    response_schemas = [
        ResponseSchema(name="query_type", description="Query types in user questions,"
                                                      "Example:"
                                                      "User question: Check the detailed specifications"
                                                      "Type: query"
                                                      "User question: Can you please provide detailed statistics on the specifications"
                                                      "Type: count"
                                                      "User question: Take a look at the detailed specifications"
                                                      "Type: query"
                                                      "User question: Based on the type of land used, please provide a detailed plan"
                                                      "Type: count"),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(
        template="You are a little expert in analyzing user problems, specializing in breaking down user problems. "
                 "No need to answer user's questions, just break down the user's questions."
                 "\n{format_instructions}\n{question}",
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions},
    )
    _input = prompt.format_prompt(question=query)

    # output = model(_input.to_string())
    output = model(_input.to_messages())
    json_obj = None
    try:
        # json_str_for_markdown = f"```json\n{output.content}\n```"
        json_obj = parse_json_markdown(output.content)
    except json.JSONDecodeError as e:
        json_obj = None
    query_type = "query"
    if json_obj is not None:
        query_type = json_obj['query_type'] if "query_type" in json_obj else ""
    return str(query_type)


def get_count_condition_by_question(model: llm, query: str) -> str:
    response_schemas = [
        ResponseSchema(name="table_info",
                       description="用户问题中可能存在的表名"                                           "例子:"
                                   "用户问题: 纽约市2020年的人口数是多少？"
                                   "表名: 纽约市"),
        ResponseSchema(name="field_info", description="用户问题中可能存在的字段名"
                                                      "例子:"
                                                      "用户问题: 纽约市2020年房屋总数量是多少？"
                                                      "字段名: 房屋总数量"),
        ResponseSchema(name="condition_info", description="用户问题中可能存在的过滤条件"
                                                          "例子1:"
                                                          "帮我根据房屋是否被出租分组统计2020年纽约市房屋总数量。"
                                                          "过滤条件: 无"
                                                          "例子2："
                                                          "帮我根据房屋大小分组统计2020年纽约市房屋总数量。"
                                                          "过滤条件: 无"
                                                          "例子3："
                                                          "帮我根据房屋大小分组统计2020年纽约市房屋总数量，过滤条件为：房屋大小大于100平方米。"
                                                          "过滤条件: 房屋大小大于100平方米"),
        ResponseSchema(name="group_by_info", description="用户问题中可能存在的分组统计条件"
                                                         "例子1:"
                                                         "帮我统计2020年纽约市房屋总数量"
                                                         "分组统计条件: 无"
                                                         "例子2:"
                                                         "帮我统计一下纽约市白人人口数量"
                                                         "分组统计条件: 无"
                                                         "例子3:"
                                                         "帮我根据房屋是否被出租分组统计2020年纽约市房屋总数量"
                                                         "分组统计条件: 房屋是否被出租"),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(
        template=
        "你是分析用户问题的专家，专门分解用户问题。"
        "现在不需要回答用户的问题，只需根据我给出的模式和示例分解用户的问题，并返回相应的短语，请用中文回答。"
        "请注意，有些条件可能不存在为“无”，只用返回“无”即可，不要编造用户问题中不存在的条件或短语"
        "\n{format_instructions}\n{question}",
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions},
    )
    _input = prompt.format_prompt(question=query)
    output = model(_input.to_messages())
    json_obj = None
    try:
        # json_str_for_markdown = f"```json\n{output.content}\n```"
        json_obj = parse_json_markdown(output.content)
    except json.JSONDecodeError as e:
        json_obj = None
    table_info = ""
    field_info = ""
    condition_info = ""
    group_by_info = ""
    if json_obj is not None:
        table_info = json_obj['table_info'] if "table_info" in json_obj else ""
        field_info = json_obj['field_info'] if "field_info" in json_obj else ""
        # 更正condition_info的判断逻辑，这里我们不检查'year'，而是直接尝试获取'condition_info'
        condition_info = json_obj['condition_info'] if "condition_info" in json_obj else ""
        group_by_info = json_obj['group_by_info'] if "group_by_info" in json_obj else ""
    if table_info == "无":
        table_info = ""
    if field_info == "无":
        field_info = ""
    if condition_info == "无":
        condition_info = ""
    if group_by_info == "无":
        group_by_info = ""

    return table_info, field_info, condition_info, group_by_info


def get_query_condition_by_question(model: llm, query: str) -> str:
    response_schemas = [
        ResponseSchema(name="table_info",
                       description="用户问题中可能存在的表名"                                           "例子:"
                                   "用户问题: 纽约市2020年的人口数是多少？"
                                   "表名: 纽约市"),
        ResponseSchema(name="field_info", description="用户问题中可能存在的字段名"
                                                      "例子:"
                                                      "用户问题: 纽约市2020年房屋总数量是多少？"
                                                      "字段名: 房屋总数量"),
        ResponseSchema(name="condition_info", description="用户问题中可能存在的过滤条件"
                                                          "例子1:"
                                                          "帮我根据房屋是否被出租分组统计2020年纽约市房屋总数量。"
                                                          "过滤条件: 无"
                                                          "例子2："
                                                          "帮我根据房屋大小分组统计2020年纽约市房屋总数量。"
                                                          "过滤条件: 无"
                                                          "例子3："
                                                          "帮我根据房屋大小分组统计2020年纽约市房屋总数量，过滤条件为：房屋大小大于100平方米。"
                                                          "过滤条件: 房屋大小大于100平方米"),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(
        template=
        "你是分析用户问题的小专家，专门分解用户问题。"
        "现在不需要回答用户的问题，只需根据我给出的模式和示例分解用户的问题，并返回相应的短语，请用中文回答。"
        "请注意，有些条件可能不存在为“无”，只用返回“无”即可，不要编造用户问题中不存在的条件或短语"
        "\n{format_instructions}\n{question}",
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions},
    )
    _input = prompt.format_prompt(question=query)
    output = model(_input.to_messages())
    json_obj = None
    try:
        # json_str_for_markdown = f"```json\n{output.content}\n```"
        json_obj = parse_json_markdown(output.content)
    except json.JSONDecodeError as e:
        json_obj = None
    table_info = ""
    field_info = ""
    condition_info = ""
    if json_obj is not None:
        table_info = json_obj['table_info'] if "table_info" in json_obj else ""
        field_info = json_obj['field_info'] if "field_info" in json_obj else ""
        # 更正condition_info的判断逻辑，这里我们不检查'year'，而是直接尝试获取'condition_info'
        condition_info = json_obj['condition_info'] if "condition_info" in json_obj else ""
    if table_info == "无":
        table_info = ""
    if field_info == "无":
        field_info = ""
    if condition_info == "无":
        condition_info = ""

    return table_info, field_info, condition_info


def get_sql_query(model, query, query_type) -> str:
    sql_query = ""
    if query_type == 'count':
        # 拆分问题中的统计字段
        # 拆分统计问题中的条件
        condition = get_count_condition_by_question(model, query)
        print(condition)
        count_table_info = condition[0]
        count_field_info = condition[1]
        count_condition_info = condition[2]
        count_group_by_info = condition[3]
        # fields = get_count_fields_by_question(model, query)
        sql_query = f"统计数据的数量"
        if count_field_info != '':
            if count_group_by_info != '':
                sql_query = f"统计该列:“{count_field_info}”,根据“{count_group_by_info}”分组统计"
            else:
                sql_query = f"统计该列:“{count_field_info}”的和"
        if count_group_by_info != '':
            if count_field_info != '':
                sql_query = f"统计该列:“{count_field_info}”的和,根据“{count_group_by_info}”分组统计"
            else:
                sql_query = f"根据“{count_group_by_info}”分组统计"
        if count_condition_info == '':
            sql_query += '，无需过滤条件'
        else:
            sql_query += f",过滤条件：{count_condition_info}"
    else:
        condition = get_query_condition_by_question(model, query)
        count_table_info = condition[0]
        count_field_info = condition[1]
        count_condition_info = condition[2]
        sql_query = f"查询{count_table_info}数据"
        if count_field_info != '':
            sql_query = f"查询{count_table_info}数据,只查这几列：{count_field_info}"
        count_condition_info = '无' if count_condition_info == '' else count_condition_info
        sql_query += f",查询条件：{count_condition_info}"
    return sql_query

from langchain.prompts import PromptTemplate,ChatPromptTemplate
from langchain.chains import LLMChain

def get_llm_sql(table_name: str, field_info_list: list[dict], query: str, model) -> str:
    prompt_template = PromptTemplate(
        input_variables=["table_name", "field_list_str", "question"],
        template="你是SQL的专家。给定一个输入问题，创建一个语法正确的SQL语句，确保标点符号正确表示。\n"
                 "注意：请根据已知表名称和列名称生成sql语句！不要生成未知的列名称！只返回sql即可，不要编造多余的话语\n"
                 "注意:用户给的问题一般都比较简单，不涉及复杂的sql语句查询，按用户的问题需求生成简洁有效的SQL语句即可，不要乱加AND或者WHERE的过滤条件\n"
                 "注意：当用户的问题中没有过滤条件时，不要乱加用户问题中没有的东西作为过滤条件\n"
                 "已知数据表名称：”{table_name}“\n"
                 "已知列名称与描述集合：\n"
                 "{field_list_str}\n"
                 "输入问题：”{question}“\n"
    )
    if table_name == '':
        return ''
    if len(field_info_list) < 1:
        return ''
    chain = LLMChain(prompt=prompt_template, llm=model)
    field_list = []
    for field_info in field_info_list:
        field_name = field_info.get("field_name")
        field_desc = field_info.get("field_desc")
        field_list.append(f'列名：“{field_name}“,描述：“{field_desc}“')
    field_list_str = "\n".join(field_list)
    r = chain.invoke({"table_name": table_name, "field_list_str": field_list_str, "question": query})
    print(r['text'])
    print("=====================")
    llm_sql = ""
    result_text = r['text']
    if '```sql' in result_text:
        import re

        # 正则表达式模式，匹配以 ```sql 或 ~~~sql 开始，至对应的结束标记 ``` 或 ~~~ 之间的内容
        pattern = r'```sql[\s\S]*?```|~~~sql[\s\S]*?~~~'
        # 使用 re.findall 找出所有匹配的SQL代码块
        sql_blocks = re.findall(pattern, result_text)
        # 去除代码块标记，仅保留SQL代码
        cleaned_sql_blocks = [
            block.replace('```sql', '').replace('```', '').replace('~~~sql', '').replace('~~~', '') for
            block
            in sql_blocks]
        cleaned_sql = cleaned_sql_blocks[0]
        llm_sql = cleaned_sql
    else:
        llm_sql = result_text

    # llm_sql = result_text
    # llm_sql = format_sql(llm_sql, field_info_list)
    return llm_sql


import psycopg2


def query_data(sql_query):
    # 数据库连接配置，请根据实际情况填写
    connection_config = {
        "dbname": "postgres",
        "user": "sd_test",
        "password": "sd_test@ai",
        "host": "168.4.4.61",
        "port": "5432"
    }

    try:
        # 连接到数据库
        conn = psycopg2.connect(**connection_config)
        cursor = conn.cursor()

        # 执行SQL查询
        cursor.execute(sql_query)


        # 获取查询结果
        query_result = cursor.fetchall()

        # 关闭游标和连接
        cursor.close()
        conn.close()

        # 将查询结果转化为JSON格式（这里简单处理，实际可能需要更复杂的转换逻辑）
        # 注意：这里的转换假设结果是简单的，如数字或单层列表
        # 对于复杂结构，可能需要使用json.dumps加上适当的序列化逻辑
        # json_result = json.dumps(query_result)
        # print(json_result)
        return query_result

    except psycopg2.Error as e:
        # 捕获数据库相关的错误
        print(f"数据库错误：{e}")
        return None
    except Exception as e:
        # 捕获其他异常
        print(f"发生错误：{e}")
        return None


def get_custom_question(model, query: str) -> dict:
    try:
        response_schemas = [
            ResponseSchema(name="year", description="...")
        ]
        # 设置解析器等
        output = model(query)
        json_obj = parse_json_markdown(output.content)

        year_info = json_obj.get('year', '')
        table_info = json_obj.get('table_info', '')

        # 返回解析结果
        return str(year_info), str(table_info)
    except Exception as e:
        print(f"模型调用错误：{e}")
        return '', ''  # 默认返回空值


import pandas as pd
import re
import time

# 读取文件
try:
    df = pd.read_excel('/root/langchat_3dgis/table/exp1/QA.xlsx')
except Exception as e:
    print(f"读取文件时发生错误：{e}")
    df = pd.DataFrame()  # 返回一个空的 DataFrame，避免后续出错

# SQL 关键字的正则表达式
sql_keywords_regex = r'\b(SELECT|SUM|MIN|MAX|AS|COUNT|FROM|WHERE|JOIN|ON|GROUP BY|ORDER BY|HAVING|LIMIT)\b'

# 提取 SQL 语句中的关键字
def extract_keywords(sql):
    return re.findall(sql_keywords_regex, sql, re.IGNORECASE)

# 错误处理输出对象
class AgentOutput:
    def __init__(self, success: bool, result=None, error=None):
        self.success = success
        self.result = result
        self.error = error

# Agent 1 的实现
def agent_1_process(model, query):
    try:
        year, table_info = get_custom_question(model, query)
        if not table_info:
            raise ValueError("未能从用户问题中提取表名信息")
        return AgentOutput(success=True, result={"year": year, "table_info": table_info})
    except Exception as e:
        return AgentOutput(success=False, error=f"Agent 1 错误: {str(e)}")

# Agent 2 的实现
def agent_2_process(model, query):
    try:
        query_type = get_type_by_question(model, query)
        if not query_type:
            raise ValueError("未能从用户问题中提取查询类型")
        return AgentOutput(success=True, result={"query_type": query_type})
    except Exception as e:
        return AgentOutput(success=False, error=f"Agent 2 错误: {str(e)}")

# Agent 3 的实现
def agent_3_process(model, query):
    try:
        count_condition = get_count_condition_by_question(model, query)
        if not count_condition:
            raise ValueError("未能从用户问题中提取计数条件")
        return AgentOutput(success=True, result={"count_condition": count_condition})
    except Exception as e:
        return AgentOutput(success=False, error=f"Agent 3 错误: {str(e)}")

# Agent 4 的实现
def agent_4_process(model, query):
    try:
        query_condition = get_query_condition_by_question(model, query)
        if not query_condition:
            raise ValueError("未能从用户问题中提取查询条件")
        return AgentOutput(success=True, result={"query_condition": query_condition})
    except Exception as e:
        return AgentOutput(success=False, error=f"Agent 4 错误: {str(e)}")

# Agent 5 的实现
def agent_5_process(model, query):
    try:
        sql_query = get_llm_sql(model, query)
        if not sql_query:
            raise ValueError("未能生成 SQL 查询")
        return AgentOutput(success=True, result={"sql_query": sql_query})
    except Exception as e:
        return AgentOutput(success=False, error=f"Agent 5 错误: {str(e)}")

# Agent 输出是否符合预期
def validate_agent_output(agent_output, validation_rules):
    if not agent_output.success:
        return False, agent_output.error

    for rule in validation_rules:
        if not rule(agent_output.result):
            return False, "输出未通过验证规则"

    return True, None

# 完整查询处理流程
# 完整查询处理流程
def process_query_with_agents(query, model, rollback_agent=None):
    """
    递归地处理每个 Agent，遇到错误时回退并重新处理前一个 Agent。
    """
    # Step 1: Agent 1 处理表名和年份
    if rollback_agent is None or rollback_agent == 1:
        agent_1_output = agent_1_process(model, query)
        if not agent_1_output.success or not agent_1_output.result.get("table_name") or not agent_1_output.result.get("year"):
            print(f"Agent 1 失败，错误信息: {agent_1_output.error}")
            # 如果需要回退到上一个 Agent，可以传递参数，控制递归回退
            return process_query_with_agents(query, model, rollback_agent=0)  # 回退至初始阶段

    # Step 2: Agent 2 获取查询类型
    if rollback_agent is None or rollback_agent == 2:
        agent_2_output = agent_2_process(model, query)
        if not agent_2_output.success or not agent_2_output.result.get("query_type"):
            print(f"Agent 2 失败，错误信息: {agent_2_output.error}")
            return process_query_with_agents(query, model, rollback_agent=0)  # 回退到 Agent 1

    # Step 3: 根据查询类型调用 Agent 3 或 Agent 4
    if rollback_agent is None or rollback_agent == 3:
        if agent_2_output.result["query_type"] == "count":
            agent_3_output = agent_3_process(model, query)
            if not agent_3_output.success or not agent_3_output.result.get("conditions"):
                print(f"Agent 3 失败，错误信息: {agent_3_output.error}")
                return process_query_with_agents(query, model, rollback_agent=2)  # 回退到 Agent 2
        else:
            agent_4_output = agent_4_process(model, query)
            if not agent_4_output.success or not agent_4_output.result.get("conditions"):
                print(f"Agent 4 失败，错误信息: {agent_4_output.error}")
                return process_query_with_agents(query, model, rollback_agent=2)  # 回退到 Agent 2

    # Step 4: 调用 Agent 5 生成 SQL
    if rollback_agent is None or rollback_agent == 4:
        agent_5_output = agent_5_process(model, query)
        if not agent_5_output.success or not agent_5_output.result.get("sql_query"):
            print(f"Agent 5 失败，错误信息: {agent_5_output.error}")
            return process_query_with_agents(query, model, rollback_agent=0)  # 回退到 Agent 3

    return {"sql_query": agent_5_output.result}



# 主流程
for index, row in df.iterrows():
    user_query = row['自然语言查询语句']
    print(f"查询语句: {user_query}")
    true_sql = row['对应的SQL语句']

    # 记录开始时间
    start_time = time.time()

    # 查询流程：从 Agent 1 到 Agent 5
    result = process_query_with_agents(user_query, llm)
    if "error" in result:
        print(f"查询失败: {result['error']}")
        continue  # 跳过当前行，继续下一个查询

    # 获取生成的 SQL
    generated_sql = result["sql_query"]
    print(f"生成的 SQL: {generated_sql}")

    # 提取关键字
    generated_keywords = extract_keywords(generated_sql)
    correct_keywords = extract_keywords(true_sql)

    # 计算匹配概率
    match_probability = len(set(generated_keywords) & set(correct_keywords)) / len(set(correct_keywords))

    # 输出匹配情况
    print(f"Generated SQL Keywords: {generated_keywords}")
    print(f"Correct SQL Keywords: {correct_keywords}")
    print(f"Match Probability: {match_probability:.2f}")

    # 记录结束时间
    end_time = time.time()
    print(f"处理时间: {end_time - start_time:.2f}秒")
