<template>
    <div class="flex flex-col">
        <div class="border-b-2 border-black mb-2 w-90%"></div>
        <div class="pl-2 pr-2">
            <h1 class="my-auto w-ful color-main">
                {{ props.title }}
            </h1>
            <h2>{{ props.venue }}</h2>
            <div class="mt-4">
                <p class="m-0">{{ props.author }}</p>
                <p class="m-0">
                    <a :href="`mailto:${props.email}`">{{ props.email }}</a>
                </p>
                <p class="m-0">{{ props.group }}</p>
                <p class="m-0">{{ props.affiliation }}</p>
            </div>
        </div>
        <div class="border-b-2 border-black mt-2 w-90%"></div>

    </div>
</template>

<script setup lang="ts">

const props = defineProps<{
    title: string;
    author: string;
    email: string;
    group: string;
    venue: string;
    date: string;
    affiliation: string;

}>();

</script>

<style scoped>
/* style h1 blue */
.slidev-layout.default h1 {
    color: #003d86;
}

.slidev-layout p {
    margin: 0;
}
</style>
