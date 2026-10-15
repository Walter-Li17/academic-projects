"""Generate the October 2026 CV. Requires reportlab; uses only public CV content."""
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors

O = Path(__file__).resolve().parent
styles={
 'name':ParagraphStyle('name',fontName='Helvetica-Bold',fontSize=22,leading=25,spaceAfter=5),
 'meta':ParagraphStyle('meta',fontName='Helvetica',fontSize=9,leading=12,spaceAfter=3),
 'sec':ParagraphStyle('sec',fontName='Helvetica-Bold',fontSize=11,leading=14,spaceBefore=6,spaceAfter=3,textColor=colors.HexColor('#173D47')),
 'body':ParagraphStyle('body',fontName='Helvetica',fontSize=9.2,leading=11.5,spaceAfter=2),
 'bullet':ParagraphStyle('bullet',fontName='Helvetica',fontSize=9.2,leading=11.5,leftIndent=10,firstLineIndent=-8,spaceAfter=1.5),
}
def p(t,s='body'):return Paragraph(t,styles[s])
a=[p('Ruoshui (Walter) Li','name'),p('Department of Mathematics and Statistics, Saint Louis University','meta'),p('ruoshuiliwalter.li@slu.edu | liruoshui0710@gmail.com','meta'),p('<link href="https://ruoshui-li.liruoshui0710.chatgpt.site">ruoshui-li.liruoshui0710.chatgpt.site</link> | <link href="https://github.com/Walter-Li17/academic-projects">GitHub academic portfolio</link>','meta')]
def sec(t):a.append(p(t,'sec'))
def ent(title,date,detail=''):
 t=Table([[p('<b>'+title+'</b>'),p(date)]],colWidths=[359,151],hAlign='LEFT')
 t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0),('TOPPADDING',(0,0),(-1,-1),0),('BOTTOMPADDING',(0,0),(-1,-1),0)]))
 a.append(t)
 if detail:a.append(p(detail))
def bullet(t):a.append(p('• '+t,'bullet'))

sec('Education')
ent('Saint Louis University','Fall 2025 - Present','M.A. in Mathematics, expected Spring 2027')
ent('University of Connecticut','Sept. 2020 - Dec. 2024','B.A. in Mathematics')
ent('East China Normal University','Sept. 2020 - Dec. 2021','Study abroad program during the COVID-19 pandemic')

sec("Master's Thesis (in progress)")
a.append(p('<b>Image Retrieval with DINO Features and SVD</b> (working title)'))
a.append(p('Supervisor: Vahan Huroyan | Expected completion: Spring 2027'))
bullet('Conducted experiments with frozen DINO features, comparing whole-image, matrix, and SVD-based subspace representations for retrieval on Revisited Oxford using mean average precision; evaluated one fixed candidate on Revisited Paris and visualized rankings in an interactive demo.')
bullet('Next phase: reproduce the supervised Grassmann pooling framework of Wei et al. (ECCV 2018), including learned dimensionality reduction, SVD-based pooling, and a learned classifier based on projection similarity.')

sec('Teaching Experience')
ent('Precalculus Instructor (MATH 1400)', 'Spring &amp; Fall 2026', 'Saint Louis University | Two semesters; Fall 2026 ongoing')
bullet('Introduce mathematical terminology and definitions, then use examples to help students understand the concepts.')
bullet('Prepare and refine lecture notes, worksheets, quizzes, examinations, and solutions; align assessments with taught material and classroom pacing.')
bullet('Use AI to assist with rubric-based exam grading and preparation of student feedback.')
a.append(p('<b>Teaching materials:</b> handwritten classroom notes, typed notes, worksheets, and sample assessments <link href="https://github.com/Walter-Li17/academic-projects/tree/main/teaching" color="#173D47">on GitHub</link>.','meta'))
ent('Calculus I Teaching Assistant', 'Fall 2025', 'Saint Louis University')
bullet('Led discussion sessions, explained worked problems, and graded quizzes.')

sec('Selected Academic Projects')
ent("Multivariable Taylor's Theorem", '2026')
bullet("Class presentation for MATH 5023 on directional derivatives, multivariable Taylor's theorem, proof by reduction to one variable, and applications to error estimates and critical points.")
ent('Calculating Normal Probabilities without Tables', 'Fall 2025')
bullet("Studied Bagby's 1995 approximation method using double integrals and polar coordinates.")
bullet('Gave a class presentation for MATH 5021 and a poster presentation in the Mentoring Math Minds (MMM) program at Saint Louis University.')

sec('Skills')
a.append(p('<b>Programming:</b> Python, R.'))
a.append(p('<b>AI-assisted problem solving:</b> Task decomposition, constraint setting, and iterative review.'))
bullet('<b>Research:</b> Broke a retrieval study into nine key stages; specified resource constraints and candidate choices to guide AI-assisted testing of all planned combinations.')
bullet('<b>Grading:</b> Structured rubric-based workflows with sample review checkpoints, matched feedback files, and escalation of ambiguous cases.')
bullet('<b>Teaching materials:</b> Directed revisions to notes and assessments, checking content coverage, worked solutions, and classroom pacing.')
bullet('<b>Application planning:</b> Specified a workflow linking program requirements, shared questions, and school-specific essays, with review before submission.')
a.append(Spacer(1,4))
a.append(p('<b>Additional background:</b> Completed four years at Sacred Heart High School, Waterbury, Connecticut; graduated in 2020.','meta'))
a.append(Spacer(1,2))
a.append(p('Updated October 2026','meta'))
target=O/'Ruoshui_Li_Resume.pdf'
SimpleDocTemplate(str(target),pagesize=(612,792),leftMargin=45,rightMargin=45,topMargin=28,bottomMargin=25,title='Ruoshui Li Curriculum Vitae',author='Ruoshui Li').build(a)
print(target)
