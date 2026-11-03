// Fixed, human-authored translations for display. Never used in model requests.
export const starCasesEs: Record<string, {questionDisplay: string; optionsDisplay: [string, string, string, string]}> = {
  Sequence_T3_4937: {
    questionDisplay: '¿Qué ocurrió después de que la persona se sentara en el sofá?',
    optionsDisplay: ['Tomó el teléfono/cámara.', 'Ordenó la ropa.', 'Arrojó la escoba.', 'Lavó la ropa.'],
  },
  Sequence_T4_3292: {
    questionDisplay: '¿Qué ocurrió antes de que la persona abriera la bolsa?',
    optionsDisplay: ['Ordenó el armario/gabinete.', 'Dejó la toalla.', 'Se sentó en el sofá.', 'Arrojó la almohada.'],
  },
  Sequence_T3_6255: {
    questionDisplay: '¿Qué ocurrió después de que la persona se acostara en la cama?',
    optionsDisplay: ['Ordenó el armario/gabinete.', 'Abrió el armario/gabinete.', 'Tomó el sándwich.', 'Abrió la bolsa.'],
  },
  Interaction_T1_262: {
    questionDisplay: '¿Qué objeto tomó la persona?',
    optionsDisplay: ['La toalla.', 'El sándwich.', 'El papel/cuaderno.', 'El cuadro.'],
  },
  Interaction_T4_572: {
    questionDisplay: '¿Qué hizo la persona mientras se apoyaba en la silla y llevaba la caja?',
    optionsDisplay: ['Tomó el papel/cuaderno.', 'Se sentó en el sofá.', 'Lavó la ropa.', 'Se acostó en el sofá.'],
  },
  Interaction_T1_7248: {
    questionDisplay: '¿Qué objeto dejó la persona?',
    optionsDisplay: ['La ropa.', 'El sándwich.', 'La comida.', 'La bolsa.'],
  },
  Prediction_T4_634: {
    questionDisplay: '¿Qué objeto abriría la persona a continuación, después de cerrar la ventana?',
    optionsDisplay: ['La bolsa.', 'La puerta.', 'El armario/gabinete.', 'El libro.'],
  },
  Prediction_T1_1408: {
    questionDisplay: '¿Qué hará la persona a continuación?',
    optionsDisplay: ['Tomar la taza/vaso/botella.', 'Dejar la escoba.', 'Abrir la bolsa.', 'Acostarse en el sofá.'],
  },
  Feasibility_T5_305: {
    questionDisplay: '¿Qué objeto puede tomar la persona después de agarrar el pomo de la puerta?',
    optionsDisplay: ['El libro.', 'La bolsa.', 'El zapato.', 'El teléfono/cámara.'],
  },
  Feasibility_T5_2170: {
    questionDisplay: '¿Qué objeto puede arrojar la persona después de atravesar la puerta?',
    optionsDisplay: ['La escoba.', 'La bolsa.', 'La ropa.', 'La manta.'],
  },
};
export function presentCase<T extends {id: string; question: string; options: string[]}>(record: T) {
  const display = starCasesEs[record.id];
  return {...record, questionOriginal: record.question, optionsOriginal: record.options,
    questionDisplay: display?.questionDisplay || 'Traducción de la pregunta no disponible.',
    optionsDisplay: display?.optionsDisplay || record.options.map(() => 'Traducción no disponible.')};
}
export function answerDisplay(caseId: string, letter: string) {
  return starCasesEs[caseId]?.optionsDisplay['ABCD'.indexOf(letter)] || 'Traducción no disponible.';
}
