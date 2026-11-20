from pydantic import BaseModel, Field, model_validator

# OJO: el docstring de la clase y las description de los campos se le mandan al LLM como parte
# de las instrucciones de formato (PydanticOutputParser.get_format_instructions()).
# Por eso están escritos pensando en el modelo, y las notas para desarrolladores van en comentarios #.

RESPUESTA_DESCONOCIDA = "No lo sé"


class RespuestaRAG(BaseModel):
    """Respuesta a la pregunta del usuario, basada únicamente en el contexto recuperado."""

    respuesta: str = Field(
        description=(
            "Respuesta en español, usando solo información del contexto. "
            f"Si el contexto no alcanza para responder, exactamente: '{RESPUESTA_DESCONOCIDA}'."
        ),
    )
    referencias: list[str] = Field(
        default_factory=list,
        description=(
            "Ids de los fragmentos del contexto usados para responder, tal como aparecen entre corchetes "
            "(por ejemplo 'como_llegar.md#0'). Lista vacía si la respuesta no está en el contexto."
        ),
    )
    encontrada: bool = Field(
        description="true si la respuesta está en el contexto; false si no lo está.",
    )

    # Corre después de validar cada campo por separado y controla que sean coherentes entre sí.
    # Un modelo que dice haber encontrado la respuesta pero no cita ninguna fuente
    # probablemente la sacó de su propio conocimiento, que es justo lo que se quiere evitar.
    @model_validator(mode="after")
    def validar_coherencia(self) -> "RespuestaRAG":
        if self.encontrada and not self.referencias:
            raise ValueError("Si la respuesta está en el contexto, tiene que citar al menos una referencia")
        if not self.encontrada:
            # Sin respuesta no hay fuentes: se normaliza para que la salida sea siempre la misma
            self.respuesta = RESPUESTA_DESCONOCIDA
            self.referencias = []
        return self
