from pathlib import Path
from lxml import etree

SRC_DIR = Path(__file__).resolve().parent
POSTS_XML = SRC_DIR.parent / "data" / "raw" / "security-stackexchange" / "Posts.xml"

POST_HISTORY_XML = POSTS_XML.parent / "PostHistory.xml"

questions = {}
answers = []

def clean_tags(raw_tag):
    tags = [t for t in raw_tag.split("|") if t]
    return tags

def load_close_reasons():
    latest = {}
    for event, element in etree.iterparse(str(POST_HISTORY_XML), events=("end",), tag="row"):
        if element.get("PostHistoryTypeId") == "10":
            q_id, date = element.get("PostId"), element.get("CreationDate")
            if q_id not in latest or date > latest[q_id][0]:
                latest[q_id] = (date, element.get("Comment"))
        element.clear()
    return {q_id: reason for q_id, (date, reason) in latest.items()}

close_reasons = load_close_reasons()

for event, element in etree.iterparse(str(POSTS_XML), events = ("end",), tag ="row"):
    post_type = element.get("PostTypeId")
    if post_type == "1":
        q_id = element.get("Id")
        questions[q_id] = {
            "creation_date":element.get("CreationDate"),
            "score":element.get("Score"),
            "view_count":element.get("ViewCount"),
            "title":element.get("Title"),
            "body":element.get("Body"),
            "tags":clean_tags(element.get("Tags")),
            "answer_count":element.get("AnswerCount"),
            "accepted_answer_id":element.get("AcceptedAnswerId"),
            "last_activity_date":element.get("LastActivityDate"),
            "owner_user_id":element.get("OwnerUserId"),
            "content_license": element.get("ContentLicense"),
            "close_reason": close_reasons.get(q_id) if element.get("ClosedDate") else None,

        }

    elif post_type == "2":
        answers.append({
            "id" : element.get("Id"),
            "creation_date":element.get("CreationDate"),
            "score":element.get("Score"),
            "body":element.get("Body"),
            "last_activity_date":element.get("LastActivityDate"),
            "parent_id":element.get("ParentId"),
            "content_license": element.get("ContentLicense")
        })

    element.clear()  

orphans = [a for a in answers if a["parent_id"] not in questions]

print(len(questions), "questions,", len(answers), "answers,", len(orphans), "orphans")
