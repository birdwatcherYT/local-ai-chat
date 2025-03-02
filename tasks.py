from invoke import task


@task
def chat(c):
    """AIとのチャット"""
    from chat import chat_start

    chat_start(c.config)


@task
def vv_list(c):
    """VOICEVOXの一覧を表示"""
    from voicevox import print_speakers

    print_speakers()


@task
def ci_list(c):
    """COEIROINKの一覧を表示"""
    from coeiroink import print_speakers

    print_speakers()
