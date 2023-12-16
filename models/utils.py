def make_prompt(techniques=[]):
    '''
    Apply prompt techniques (several tehniques are possible, even all of them) to a template.
    Possible techniques values: "cot", "reit", "right", "rawinst", "pos"
    https://arxiv.org/pdf/2303.07142.pdf
    '''
    template = (
        "Vous faites partie du système de questions-réponses RAG. Vous avez une question et des paragraphes qui s'y rapportent. Chaque paragraphe se termine par la chaîne \"Media ID\". Répondez à la question en utilisant ces paragraphes. Après votre réponse, écrivez \"Media ID\" des paragraphes que vous avez utilisés pour obtenir la réponse dans le champ \"media_id\". \n"
        "Paragraphes pertinents.\n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Repond la question: {query_str}.\n"
        "Vous devez répondre à la question au format JSON: {{\"answer\": \"<votre réponse>\", \"media_id\": \"<identifiant de média utilisé>\"}}. Vous devez inclure tous les mots de votre réponse en JSON uniquement, pas d'autres formats.\n"
        "Si vous ne pouvez pas donner de réponse sur la base des informations fournies, vous devez toujours suivre la structure json.\n"
    )
    if "cot" in techniques:
    #     It is a zero-shot approach, meaning that it does not rely on previously given examples. 
    #     This technique is designed to encourage the model to break down its thought process, 
    #     potentially leading to more accurate and explainable decisions.
        template = template.replace(
                "Repond la question:", 
                "Veuillez décomposer votre processus de réflexion étape par étape en détail avant de donner la réponse finale. "
                "Repond la question:"
            )
    if "reit" in techniques:
        # This method involves reinforcing key elements in the instructions by repeating them. 
        # The repetition is used to emphasize important aspects of the task, ensuring that the model 
        # pays closer attention to these elements.
        template = template.replace(
                "Repond la question:", 
                "N'oubliez pas qu'il est crucial d'utiliser les informations des paragraphes fournis pour répondre à la question. "
                "Concentrez-vous attentivement sur le contenu de ces paragraphes et référez-vous-y explicitement dans votre réponse. "
                "N'utilisez aucun fait non présenté dans les paragraphes. "
                "Repond la question:"
            )
    if "right" in techniques:
        #  In this approach, the model is specifically asked to reach the right conclusion. 
        #  This directive aims to focus the model's effort on accuracy and correctness in its response.
        template = template.replace(
                "Repond la question:", 
                "Assurez-vous que votre réponse est très précise, basée directement sur les informations contenues dans les paragraphes fournis. "
                "La précision et l'exactitude de votre réponse sont primordiales. Assurez-vous d'aligner étroitement votre réponse avec le contenu du paragraphe. "
                "N'utilisez aucun fait non présenté dans les paragraphes. N'utilisez pas des faits dont vous n'êtes pas sûr. "
                "Repond la question:"
            )
    if "rawinst" in techniques:
        # The model is explicitly told what its job is (e.g., an AI expert in career advice) and what it needs to do
        # (e.g., sort through jobs and decide their suitability for graduates).
        template = template.replace(
                "Vous faites partie du système de questions-réponses RAG. ", 
                "Vous faites partie du système de questions-réponses RAG. "
                "En tant qu'expert en IA dans l'analyse et la synthèse d'informations textuelles, votre tâche consiste à méticuleusement "
                "triez les paragraphes suivants et déterminez la réponse la plus précise et la plus pertinente à la question présentée. "
                
            )
    if "pos" in techniques:
        # Positive feedback is provided to the model before querying it. This technique is based on the idea
        #  that setting a positive tone or providing encouragement might positively influence the model's performance.
        template = template.replace(
                "Vous faites partie du système de questions-réponses RAG. ", 
                "Vous faites partie du système de questions-réponses RAG. "
                "Vous avez toujours fourni d'excellentes réponses. Continuez cette performance exceptionnelle "
                "en abordant la question suivante avec le même niveau de précision et d'attention aux détails. "
            )
    if "shitty-few-shot" in techniques:
        template = """
            Paragraphes pertinents:
            ---------------------
            "Lors de mon discours, j'ai souligné l'importance des énergies renouvelables et présenté nos plans pour augmenter la production d'énergie solaire d'ici 2025."Media ID: rts-ZT000000-M000
            "En plus de l'énergie solaire, nous nous concentrons également sur l'énergie éolienne, qui est un autre domaine clé de notre stratégie environnementale."Media ID: rts-ZT000000-M001
            ---------------------
            Repond la question: Sur quelles sources d'énergie renouvelables l'orateur s'est-il concentré dans son discours?
            {{
            "answer": "Énergie solaire et éolienne",
            "media_id": "rts-ZT000000-M000"
            }}

            Paragraphes pertinents:
            ---------------------
            "Le budget du prochain exercice comprend des investissements importants dans l'éducation, y compris les nouvelles technologies dans les salles de classe."Media ID: rts-ZT000000-M002
            "Nous croyons qu'investir dans l'éducation, c'est investir dans notre avenir, et cela inclut la modernisation des infrastructures scolaires."Media ID: rts-ZT000000-M003
            ---------------------
            Repond la question: Quels sont les principaux domaines d'investissement dans l'éducation selon le discours?
            {{
            "answer": "Nouvelles technologies dans les salles de classe et modernisation des infrastructures scolaires",
            "media_id": "rts-ZT000000-M003"
            }}

            Paragraphes pertinents:
            ---------------------
            "Lors de mon discours, j'ai souligné l'importance des énergies renouvelables et présenté nos plans pour augmenter la production d'énergie solaire d'ici 2025."Media ID: rts-ZT000000-M000
            ---------------------
            Repond la question: Quels sont les principaux domaines d'investissement dans l'éducation selon le discours?
            {{
            "answer": "Sur la base des paragraphes fournis, je ne peux pas répondre à la question",
            "media_id": ""
            }}

            Paragraphes pertinents:
            ---------------------
            {context_str}
            ---------------------
            Repond la question: {query_str}.
            """
    if "few-shot" in techniques:
        template = template.replace("Paragraphes pertinents.\n",
            """Paragraphes pertinents.
            Il y a plusieurs exemples.
            Paragraphes pertinents:
            ---------------------
            "Lors de mon discours, j'ai souligné l'importance des énergies renouvelables et présenté nos plans pour augmenter la production d'énergie solaire d'ici 2025."Media ID: rts-ZT000000-M000
            "En plus de l'énergie solaire, nous nous concentrons également sur l'énergie éolienne, qui est un autre domaine clé de notre stratégie environnementale."Media ID: rts-ZT000000-M001
            ---------------------
            Repond la question: Sur quelles sources d'énergie renouvelables l'orateur s'est-il concentré dans son discours?
            {{
            "answer": "Énergie solaire et éolienne",
            "media_id": "rts-ZT000000-M000"
            }}

            Paragraphes pertinents:
            ---------------------
            "Le budget du prochain exercice comprend des investissements importants dans l'éducation, y compris les nouvelles technologies dans les salles de classe."Media ID: rts-ZT000000-M002
            "Nous croyons qu'investir dans l'éducation, c'est investir dans notre avenir, et cela inclut la modernisation des infrastructures scolaires."Media ID: rts-ZT000000-M003
            ---------------------
            Repond la question: Quels sont les principaux domaines d'investissement dans l'éducation selon le discours?
            {{
            "answer": "Nouvelles technologies dans les salles de classe et modernisation des infrastructures scolaires",
            "media_id": "rts-ZT000000-M003"
            }}

            Paragraphes pertinents:
            ---------------------
            "Lors de mon discours, j'ai souligné l'importance des énergies renouvelables et présenté nos plans pour augmenter la production d'énergie solaire d'ici 2025."Media ID: rts-ZT000000-M000
            ---------------------
            Repond la question: Quels sont les principaux domaines d'investissement dans l'éducation selon le discours?
            {{
            "answer": "Sur la base des paragraphes fournis, je ne peux pas répondre à la question",
            "media_id": ""
            }}

            Maintenant, utilisez les paragraphes ci-dessous pour répondre à la question ci-dessous.
            Paragraphes pertinents:
            ---------------------
            {context_str}
            ---------------------
            Repond la question: {query_str}.
            """
        )
    return template