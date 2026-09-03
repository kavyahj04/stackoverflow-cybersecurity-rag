from lxml import etree
questions = {}
answers = []
for event, element in etree.iterparse("../data/raw/security-stackexchange/Posts.xml", events = ("end",), tag ="row"):
    post_type = element.get("PostTypeId")
    if post_type == "1":
        q_id = element.get("Id")
        questions[q_id] = {
            "creation_date":element.get("CreationDate"),
            "score":element.get("Score"),
            "view_count":element.get("ViewCount"),
            "title":element.get("Title"),
            "body":element.get("Body"),
            "tags":element.get("Tags"),
            "answer_count":element.get("AnswerCount"),
            "accepted_answer_id":element.get("AcceptedAnswerId"),
            "last_activity_date":element.get("LastActivityDate"),
            "owner_user_id":element.get("OwnerUserId"),
            "content_license": element.get("ContentLicense")
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

print(len(questions), "questions")
print(len(answers), "answers")

orphans = [a for a in answers if a["parent_id"] not in questions]
print(len(orphans), "orphaned answers out of", len(answers))