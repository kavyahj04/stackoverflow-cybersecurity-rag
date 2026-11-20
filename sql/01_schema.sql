CREATE TABLE questions (
    question_id        INTEGER PRIMARY KEY,
    title              TEXT NOT NULL,
    body               TEXT NOT NULL,
    tags               TEXT[] NOT NULL,
    score              INTEGER NOT NULL,
    view_count         INTEGER,
    accepted_answer_id INTEGER,
    creation_date      TIMESTAMP NOT NULL,
    last_activity_date TIMESTAMP,
    embedding          HALFVEC(3072) NOT NULL
);

CREATE TABLE answers (
    answer_id          INTEGER PRIMARY KEY,
    question_id        INTEGER NOT NULL REFERENCES questions(question_id) ON DELETE CASCADE,
    body               TEXT NOT NULL,
    score              INTEGER NOT NULL,
    is_accepted        BOOLEAN NOT NULL,
    creation_date      TIMESTAMP NOT NULL,
    last_activity_date TIMESTAMP
);

-- Fast lookup of "all answers for these questions"
CREATE INDEX answers_question_id_idx ON answers (question_id);
