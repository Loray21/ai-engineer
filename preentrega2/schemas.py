from enum import StrEnum

from pydantic import BaseModel, Field, SecretStr, field_validator


# OJO: los docstrings de estas clases y las description de los campos terminan dentro del
# JSON Schema que se le manda al LLM. Por eso las notas para desarrolladores van en comentarios #,
# que Pydantic no lee, y los docstrings se escriben pensando en el modelo.


# Valores de relleno que un LLM suele inventar cuando un campo obligatorio no tiene datos reales.
# Se comparan en minúscula y sin <>, [] ni (), así "<UNKNOWN>" y "Unknown" cuentan igual.
VALORES_DE_RELLENO = {
    "unknown", "desconocido", "desconocida", "ninguna", "ninguno", "none", "null",
    "n/a", "na", "-", "no especificado", "no aplica", "sin tecnologías",
}


# StrEnum: cada opción es también un string ("alta" == NivelCriticidad.ALTA).
# En el JSON Schema aparece como {"enum": ["baja", "media", "alta"]},
# así el LLM sabe que no puede inventar otro valor como "urgente".
class NivelCriticidad(StrEnum):
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


# El contrato de salida del pipeline. Las restricciones (min_length, el enum, el validador)
# las controla Pydantic de nuestro lado, después de que el LLM responde.
class ExtraccionTecnica(BaseModel):
    """Entidades técnicas extraídas de un texto (descripción de arquitectura o log de error)."""

    tecnologias: list[str] = Field(
        min_length=1,
        description=(
            "Tecnologías concretas mencionadas en el texto: lenguajes, frameworks, bases de datos, "
            "servicios cloud, herramientas. Solo nombres propios, sin conceptos genéricos como 'API' o 'caché'."
        ),
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        description="Qué tan grave o urgente es la situación descrita en el texto.",
    )
    resumen_tecnico: str = Field(
        min_length=20,
        max_length=400,
        description="Resumen técnico de una o dos oraciones: qué componentes hay y cuál es el problema o el objetivo.",
    )

    @field_validator("tecnologias")
    @classmethod
    def limpiar_tecnologias(cls, valores: list[str]) -> list[str]:
        """
        Corre DESPUÉS de verificar que es una lista de strings.
        Saca espacios, descarta vacíos y valores de relleno como "<UNKNOWN>", y elimina duplicados
        ("Redis" y "redis" cuentan como uno).
        Si después de limpiar no queda nada, falla: así ["", " "] o ["<UNKNOWN>"] no pasan el min_length.
        """
        limpias: list[str] = []
        vistas: set[str] = set()
        for valor in valores:
            nombre = valor.strip()
            clave = nombre.strip("<>[]()").strip().lower()
            if not clave or clave in VALORES_DE_RELLENO or clave in vistas:
                continue
            vistas.add(clave)
            limpias.append(nombre)
        if not limpias:
            raise ValueError("La lista de tecnologías no puede quedar vacía ni tener solo valores de relleno")
        return limpias


# Configuración validada antes de crear el modelo.
# Esta clase no viaja al LLM, así que acá el docstring es para desarrolladores.
class LLMConfig(BaseModel):
    """
    provider: nombre registrado en chain.PROVIDERS, por ejemplo "openai".
    SecretStr enmascara la key al imprimir o loguear la config ('**********').
    """
    provider: str
    api_key: SecretStr
    model: str | None = None                      # None usa el modelo por defecto del proveedor
    max_tokens: int = Field(default=1000, gt=0)   # tope de la respuesta: si se alcanza, la salida queda cortada
