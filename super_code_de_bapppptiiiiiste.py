import json
from typing import Any, Dict
from pydantic import BaseModel, create_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI

# Stockage global de l'historique
conversation_history = []

def load_filters_model_from_json(path: str) -> type[BaseModel]:
    with open(path, 'r') as f:
        sample = json.load(f)

    fields: Dict[str, Any] = {}
    for key, value in sample.items():
        typ = type(value)
        if typ is float:
            field_type = (float, ...)
        elif typ is int:
            field_type = (int, ...)
        elif typ is bool:
            field_type = (bool, ...)
        else:
            field_type = (str, ...)
        fields[key] = field_type

    return create_model("MovieFilters", **fields)

def init():
    MovieFilters = load_filters_model_from_json("filters_schema.json")

    llm = ChatMistralAI(
        model_name="mistral-medium-latest",
        api_key="1wy0kbk1f9I7EyyV2ar6S9ZiDZ3h622B"
    )

    prompt = ChatPromptTemplate.from_template(
        """Tu es un assistant qui aide à définir les bons filtres pour rechercher un film sur TMDB.
Voici l'historique de la conversation (avec l'utilisateur) :
{user_input}

Réponds uniquement en JSON, en respectant les mêmes clés que ce schéma JSON :
{schema}
"""
    )

    # On pré-remplit le schéma avec les clés du modèle pour guider le modèle
    return prompt.partial(schema=", ".join(MovieFilters.model_fields.keys())) | llm.with_structured_output(MovieFilters)

def repond(question: str):
    global conversation_history
    chain = init()

    # Ajout de la nouvelle question à l'historique
    conversation_history.append(f"Utilisateur: {question}")

    # Concaténation de l'historique complet pour fournir le contexte
    context = "\n".join(conversation_history)

    # Appel du modèle avec le contexte complet
    result = chain.invoke({"user_input": context})

    # Ajout de la réponse du modèle à l'historique
    conversation_history.append(f"Assistant: {result}")

    return result

if __name__ == "__main__":
    # Exemple d'échange
    print(repond("Je veux un film dramatique français long et bien noté"))
    print(repond("Je veux aussi qu'il soit sorti après 2010"))