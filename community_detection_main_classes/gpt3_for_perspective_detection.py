import os
import openai, tiktoken, sys


openai.api_key = "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"

# few_shot_text = "Create a phrase level summary that describes the perspective in the text\n[TEXT]: The Second Amendment is not an unlimited right to own guns., The militia mentioned in the Second Amendment would have been composed of almost all adult men and, in turn, that most adult men should not have their right to own firearms infringed.,  The Second Amendment of the US Constitution protects individual gun ownership.\n[OUTPUT]: Arguments involving Second Amendment \n###\n[TEXT]: 1. More gun control laws would reduce gun deaths., While gun ownership doubled in the twentieth century, the murder rate decreased., The presence of a gun makes a conflict more likely to become violent.\n[OUTPUT]: Impact on Crime Rate\n###\n[TEXT]: Strict gun control laws do not work in Mexico, and will not work in the US., Countries with restrictive gun control laws have lower gun homicide and suicide rates than the United States.\n[OUTPUT]: Impact of Gun Control Laws in Other Countries\n###\n[TEXT]: Using tablets is more expensive than using print textbooks., E-textbooks on tablets cost less than print textbooks., A broken tablet requires an experienced technician to fix, which can be costly and time-consuming.\n[OUPUT]: Cost based Analysis between textbooks and tablets\n###\n"

#topics: vaccines, abortion, climate change, tablet vs textbook, college degree

# few_shot_sample_1 = "[TEXT]: Vaccines can save children’s lives., The American Academy of Pediatrics states that “most childhood vaccines are 90%-99% effective in preventing disease.”, Vaccines can cause serious and sometimes fatal side effects. ,According to the CDC, all vaccines carry a risk of a life-threatening allergic reaction (anaphylaxis) in about one per million children. \n[OUTPUT]: Impact of vaccine on children's lives"
#
# few_shot_sample_2 = "[TEXT]: But if at the time, she really didn't want to have me, then I say it's her choice., it should be entirely the mother's choice, not society's., What I do with my body doesn't affect the quality of their lives., I don't believe anyone should be  forced  to have an abortion it's their  choice ., it's the woman's decision , Even discounting these things it's still a woman's choice, human beings are given the right to kill others., She has control of HER body, not the fetuses., Does she not have the right to disconnect it if she feels she is not up to going through with it?\n[Output]: Abortion is a woman's right"
#
# few_shot_sample_3 = "[TEXT]: Overwhelming scientific consensus finds human activity primarily responsible for climate change., Many scientists disagree that human activity is primarily responsible for global climate change., A paper published in Asia-Pacific Journal of Atmospheric Sciences found that some climate models overstated how much warming would occur from additional C02 emissions., According to a study published in the Journal of Atmospheric and Solar-Terrestrial Physics, 50-70% of warming throughout the 20th century could be associated with an increased amount of solar activity.\n[Output]: Scientific studies regarding climate change"
#
# few_shot_sample_4 = "[TEXT]: Unlike tablets, there is no chance of getting malware, spyware, or having personal information stolen from a print textbook., On a tablet, e-textbooks can be updated instantly to get new editions or information., Tablets can hold hundreds of textbooks on one device, plus homework, quizzes, and other files, eliminating the need for physical storage of books and classroom materials.\n[Output]: Storage and security of textbook vs tablet"
#
# few_shot_sample_5 = "[TEXT]: Many college graduates are employed in jobs that do not require college degrees., College graduates have more and better employment opportunities., College graduates are more likely to have health insurance and retirement plans., Many recent college graduates are un- or underemployed., Many people succeed without college degrees.\n[Output]: Usefulness of college degrees"
#
#
# few_shot_text = "Create a phrase label that describes the perspective in the text\n" + '\n###\n'.join([few_shot_sample_1, few_shot_sample_2, few_shot_sample_3, few_shot_sample_4, few_shot_sample_5])

few_shot_sample_1 = "[TEXT]: 1. Vaccines can save children’s lives., The American Academy of Pediatrics states that “most childhood vaccines are 90%-99% effective in preventing disease.”, Vaccines can cause serious and sometimes fatal side effects. ,According to the CDC, all vaccines carry a risk of a life-threatening allergic reaction (anaphylaxis) in about one per million children. -- 2. The jew is the master of lies. Holocaust never happened., Simple arithmetic tells us that the Germans would have had to have had hundreds of camps, or else they would have had to exterminate 137 people PER HOUR, in order for six million Jews to have been exterminated at such small camps., Though six million Jews supposedly died in the gas chambers, not one body has ever been autopsied and found to have died of gas poisoning., Debating whether Holocaust happened or not is an artificial fantasy that teaches nothing except disrespect for evidence., There is so much evidence for it, it would be ridiculous to deny it., Yes the Holocaust happened. -- 3. Overwhelming scientific consensus finds human activity primarily responsible for climate change., Many scientists disagree that human activity is primarily responsible for global climate change., A paper published in Asia-Pacific Journal of Atmospheric Sciences found that some climate models overstated how much warming would occur from additional C02 emissions., According to a study published in the Journal of Atmospheric and Solar-Terrestrial Physics, 50-70% of warming throughout the 20th century could be associated with an increased amount of solar activity.  -- [OUTPUT]: Impact of vaccine on children's lives # Holocaust: fact or fiction? # Scientific studies regarding climate change"

few_shot_sample_2 = "[TEXT]: 1. But if at the time, she really didn't want to have me, then I say it's her choice., it should be entirely the mother's choice, not society's., What I do with my body doesn't affect the quality of their lives., I don't believe anyone should be  forced  to have an abortion it's their  choice ., it's the woman's decision , Even discounting these things it's still a woman's choice, human beings are given the right to kill others., She has control of HER body, not the fetuses., Does she not have the right to disconnect it if she feels she is not up to going through with it? -- [Output]: Abortion is a woman's right"

# few_shot_sample_3 = "[TEXT]: (2) Overwhelming scientific consensus finds human activity primarily responsible for climate change., Many scientists disagree that human activity is primarily responsible for global climate change., A paper published in Asia-Pacific Journal of Atmospheric Sciences found that some climate models overstated how much warming would occur from additional C02 emissions., According to a study published in the Journal of Atmospheric and Solar-Terrestrial Physics, 50-70% of warming throughout the 20th century could be associated with an increased amount of solar activity. -- [Output]: Scientific studies regarding climate change"

few_shot_sample_4 = "[TEXT]: 1. Unlike tablets, there is no chance of getting malware, spyware, or having personal information stolen from a print textbook., On a tablet, e-textbooks can be updated instantly to get new editions or information., Tablets can hold hundreds of textbooks on one device, plus homework, quizzes, and other files, eliminating the need for physical storage of books and classroom materials. -- 2. Many college graduates are employed in jobs that do not require college degrees., College graduates have more and better employment opportunities., College graduates are more likely to have health insurance and retirement plans., Many recent college graduates are un- or underemployed., Many people succeed without college degrees. -- [Output]: Storage and security of textbook vs tablet # Usefulness of college degrees"

# few_shot_sample_5 = "[TEXT]: Many college graduates are employed in jobs that do not require college degrees., College graduates have more and better employment opportunities., College graduates are more likely to have health insurance and retirement plans., Many recent college graduates are un- or underemployed., Many people succeed without college degrees. -- [Output]: Usefulness of college degrees"


few_shot_text = "Given a collection of argument clusters, the task is to create a phrase label that describes the perspective" \
                " for each cluster. The generated phrase " \
                " labels should act as summarizers of the given clusters and they should be diverse. The number" \
                " of generated labels must be exactly the same" \
                "as the number of arguments clusters." \
                "The collection will start with the token [TEXT]:.  Each cluster contains" \
                " one or more arguments, separated by a comma. Each cluster will" \
                " start with an index number," \
                " such as 1. , 2. , 3. etc and the clusters will be separated using -- followed by a newline. " \
                "If generating multiple labels, separate each label with a # sign. " \
                "Use the token '[Output]:' " \
                " at the beginning of generation. \n" + ' \n '.join([few_shot_sample_1, few_shot_sample_2, few_shot_sample_4])


def compute_number_of_tokens(text):
    encoding = tiktoken.encoding_for_model("text-davinci-003")
    tokens = encoding.encode(text)
    number_of_tokens = len(tokens)
    return number_of_tokens


# def process_output(output):
#     print("initial gpt output: ", output)
#     out = output.split('\n') # first line usually contains [Outputs]
#     ignore_line = True
#     perspective_labels = []
#     for line in out:
#         if line.startswith('1.'):
#             ignore_line = False
#         if not ignore_line:
#             l = line.split('.')[1:]
#             perspective_labels.append(''.join(l).strip())
#         else:
#             continue
#     print("perspective labels: ", perspective_labels)
#     return perspective_labels


def process_output(output, len_input_communities):
    print("initial gpt output: ", output)
    try:
        output = output.replace('[Output]:', '').replace('\n','').strip()
        perspective_labels = output.split('#')
        print("perspective labels: ", perspective_labels)
    except Exception as e:
        print("error: ", e)
        return ['Empty Label'] * len(len_input_communities)
    if len(perspective_labels) != len_input_communities:
        print("PERSPECTIVE LABELS LENGTH DO NOT MATCH THE LENGTH OF PERSPECTIVE OUTPUTS .. ")
        if len(perspective_labels) < len_input_communities:
            perspective_labels.extend(['Empty Label'] * (len_input_communities - len(perspective_labels)))
        else:
            perspective_labels = perspective_labels[:len_input_communities]

    return perspective_labels


def generate_perspective_for_arguments(all_community_args):

    # arg_str = "[TEXT]: " + ', '.join(arg_l)
    print(all_community_args[0])
    all_args_list_list = []
    all_args_list = []
    com_counter = 0
    perspectives = []
    for i, com_args in enumerate(all_community_args):
        com_args_only = [arg for _, arg, _ in com_args]
        all_args_list.append(str(com_counter + 1) + '. ' + ', '.join(com_args_only))
        com_counter += 1
        if (com_counter >= 5 and com_counter % 5 == 0) or i == len(all_community_args) - 1:
            com_counter = 0
            all_args_list_list.append(all_args_list)
            all_args_list = []
    for a_arg_list in all_args_list_list:
        all_args_str = ' -- \n'.join(a_arg_list) + ' -- ' + '\n'
        arg_str = "[TEXT]: " + '\n ' + all_args_str

        print("arg_str: ", arg_str)
        # print("few shot tokens: ", compute_number_of_tokens(few_shot_text))
        # print("prompt tokens: ", compute_number_of_tokens(arg_str))
        # return ''
        # if not the following code, GPT-3 sometimes generates a continuation text
        # if arg_str[-1] != ' ':
        #     arg_str += ' '
        response = openai.Completion.create(
          model="text-davinci-003",
          prompt=few_shot_text + '\n###\n' + arg_str,
          temperature=0,
          max_tokens=200,
          top_p=1,
          frequency_penalty=1,
          presence_penalty=1
        )

        # print(q)
        # print(response['choices'][0]['text'])
        #
        # print("....")
        # perspectives = response['choices'][0]['text'].replace('[Output]:', '').replace('\n','').strip()
        perspectives.extend(process_output(response['choices'][0]['text'], len(a_arg_list)))
    return perspectives
